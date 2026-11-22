"""Pair-level geometric constellation alignment using all source_xy context points in *_pairs.json.

Enabled with cfg["constellation_align"] = True (off when False).
An object is not a collection of independent points: when it moves closer to the camera (scale s),
shifts by (tx, ty) and rotates/flips by R (a 2D similarity transform Sim(2), 4 degrees of freedom),
all points on the source object move coherently to the target object.

How it works for one pair (training/val or test):
1. Load all K >= 3 context points `source_xy` of the pair from train_pairs.json / test_pairs.json
   (falling back to the pair's CSV query points if the JSON is absent).
2. Match all K source points on the target feature map -> initial predictions Y_raw [K, 2],
   their peak cosine similarities conf [K], and the full similarity maps sim [K, H, W] (in resized
   [0, image_size] coordinates).
3. Fit a robust 2D similarity transform T: x_src -> s * R * x_src + t (allowing reflection/flip)
   from the confident context correspondences via RANSAC + confidence-weighted Umeyama/Procrustes.
4. Add a gentle Gaussian spatial prior centered on T(x_src[k]) (prior_weight * log_prior) to each
   point's similarity map before running the windowed soft-argmax (`locate`). When the fit has too
   few inliers or extreme scale distortion, it leaves the raw prediction untouched.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.transforms import to_original, to_resized
from src.config import data_path
from src.models.matching import cell_centers, locate, sample_point_features

_PAIRS_CACHE = {}


def load_pair_source_xy(cfg, pair_id, fallback_xy=None):
    """Return float32 [K, 2] source_xy for `pair_id` from train_pairs.json or test_pairs.json."""
    data_dir = str(cfg["data_dir"])
    if data_dir not in _PAIRS_CACHE:
        table = {}
        for key in ("train_pairs_json", "test_pairs_json"):
            if not cfg.get(key):
                continue
            path = data_path(cfg, key)
            if path.is_file():
                raw = json.loads(path.read_text())
                for entry in raw.get("pairs", []):
                    pts = np.asarray(entry["source_xy"], dtype=np.float32).reshape(-1, 2)
                    table[str(entry["pair_id"])] = pts
        _PAIRS_CACHE[data_dir] = table
    pts = _PAIRS_CACHE[data_dir].get(str(pair_id))
    if pts is not None and len(pts) > 0:
        return pts.copy()
    if fallback_xy is not None:
        return np.asarray(fallback_xy, dtype=np.float32).reshape(-1, 2)
    return np.zeros((0, 2), dtype=np.float32)


def fit_similarity_2d(src_xy, trg_xy, weights=None):
    """Weighted 2D similarity transform T(x) = s * x @ R^T + t (allowing reflection).
    src_xy, trg_xy: [M, 2] numpy arrays (M >= 2). Returns (scale, R [2, 2], t [2])."""
    src = np.asarray(src_xy, dtype=np.float64)
    trg = np.asarray(trg_xy, dtype=np.float64)
    if weights is None:
        w = np.ones(len(src), dtype=np.float64)
    else:
        w = np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)
    w = w / w.sum()

    mu_s = (w[:, None] * src).sum(axis=0)
    mu_t = (w[:, None] * trg).sum(axis=0)
    ds = src - mu_s
    dt = trg - mu_t
    var_s = (w * (ds ** 2).sum(axis=1)).sum()
    if var_s < 1e-8:
        return 1.0, np.eye(2), mu_t - mu_s

    cov = (w[:, None, None] * (dt[:, :, None] * ds[:, None, :])).sum(axis=0)
    u, sv, vt = np.linalg.svd(cov)
    r = u @ vt
    scale = float(sv.sum() / var_s)
    t = mu_t - scale * (mu_s @ r.T)
    return scale, r, t


def apply_similarity_2d(xy, transform):
    scale, r, t = transform
    return scale * (np.asarray(xy, dtype=np.float64) @ r.T) + t[None, :]


def robust_constellation_fit(src_xy, trg_xy, conf, image_size, cfg):
    """RANSAC + weighted refit of a 2D similarity transform on K >= 3 correspondences in [0, image_size].
    Returns (transform, inlier_mask [K], info_dict) or (None, None, info_dict) when unreliable."""
    src = np.asarray(src_xy, dtype=np.float64)
    trg = np.asarray(trg_xy, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    k = len(src)
    min_pts = max(3, int(cfg["constellation_min_points"]))
    if k < min_pts:
        return None, None, {"applied": False, "reason": "too_few_points", "n_points": k}

    # Inlier threshold in resized pixels (e.g. 0.12 * 644 ≈ 77 px ≈ 5.5 patches)
    thresh = float(cfg["constellation_inlier_px"])
    min_scale, max_scale = float(cfg["constellation_min_scale"]), float(cfg["constellation_max_scale"])

    # Normalize confidences into positive weights via temperature softmax
    w_all = np.exp((conf - conf.max()) / 0.05)

    best_score = -1.0
    best_inliers = None
    # Deterministic subset pairs (all pairs of points (i, j) define a 2-point similarity proposal;
    # we test both non-reflected and reflected 2-point proposals plus 3-point triplets)
    indices = np.argsort(-conf)[:min(k, 12)]
    triplets = [(indices[a], indices[b], indices[c])
                for a in range(len(indices))
                for b in range(a + 1, len(indices))
                for c in range(b + 1, len(indices))]

    for trip in triplets:
        idx = list(trip)
        if np.linalg.norm(src[idx] - src[idx].mean(0)) < 14.0:
            continue
        cand = fit_similarity_2d(src[idx], trg[idx], w_all[idx])
        s, _, _ = cand
        if not (min_scale <= s <= max_scale):
            continue
        proj = apply_similarity_2d(src, cand)
        err = np.linalg.norm(proj - trg, axis=1)
        inliers = err <= thresh
        if inliers.sum() < min_pts:
            continue
        score = float((w_all[inliers] * (1.0 - 0.5 * (err[inliers] / thresh))).sum())
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum()) < min_pts:
        return None, None, {"applied": False, "reason": "no_consensus", "n_points": k}

    transform = fit_similarity_2d(src[best_inliers], trg[best_inliers], w_all[best_inliers])
    s, r, _ = transform
    if not (min_scale <= s <= max_scale):
        return None, None, {"applied": False, "reason": "scale_out_of_bounds", "n_points": k}

    det = float(np.linalg.det(r))
    angle = float(abs(math.degrees(math.atan2(r[1, 0], r[0, 0]))))
    return transform, best_inliers, {
        "applied": True,
        "n_points": k,
        "n_inliers": int(best_inliers.sum()),
        "inlier_ratio": float(best_inliers.mean()),
        "scale": float(s),
        "angle_deg": angle,
        "is_flipped": det < 0,
    }


def predict_with_constellation(src_feats, trg_feats, query_xy, all_src_xy, src_wh, trg_wh, cfg, flipped=False):
    """Predict query_xy [M, 2] using the pair's full constellation `all_src_xy` [K, 2] (in original pixels).
    Returns (pred_xy [M, 2] in original target pixels, info_dict)."""
    size, lb = cfg["image_size"], bool(cfg.get("letterbox", False))
    device = src_feats.device

    # Combine query_xy and all_src_xy; keep track of which rows correspond to query_xy
    q_orig = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    ctx_orig = np.asarray(all_src_xy, dtype=np.float32).reshape(-1, 2)
    if len(ctx_orig) == 0:
        ctx_orig = q_orig

    # Deduplicate points so each query point is matched alongside context points
    combined = list(q_orig)
    for pt in ctx_orig:
        if min(np.linalg.norm( np.asarray(combined) - pt, axis=1)) > 1.0:
            combined.append(pt)
    combined_orig = np.asarray(combined, dtype=np.float32)
    n_queries = len(q_orig)

    xy_res = to_resized(combined_orig, src_wh, size, letterbox=lb).to(device)
    if flipped:
        xy_res[:, 0] = size - xy_res[:, 0]

    desc = sample_point_features(src_feats, xy_res, size)  # [K, C]
    up_trg = trg_feats
    if cfg["feat_upsample"] > 1:
        up_trg = F.interpolate(trg_feats[None], scale_factor=cfg["feat_upsample"],
                               mode="bilinear", align_corners=False)[0]
        up_trg = F.normalize(up_trg, dim=0)

    sim = torch.einsum("kc,chw->khw", desc, up_trg)  # [K, H, W]
    raw_pred_res = locate(sim, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])  # [K, 2]
    conf = sim.flatten(1).max(dim=-1).values.detach().cpu().numpy()

    src_res_np = xy_res.detach().cpu().numpy()
    trg_res_np = raw_pred_res.detach().cpu().numpy()

    transform, _, info = robust_constellation_fit(src_res_np, trg_res_np, conf, size, cfg)
    prior_weight = float(cfg["constellation_prior_weight"])
    if transform is None or prior_weight <= 0:
        out = to_original(raw_pred_res[:n_queries].cpu(), trg_wh, size, letterbox=lb).numpy()
        return out, info

    # Geometric expectation for the M query points in resized space [M, 2]
    expected_res = torch.as_tensor(apply_similarity_2d(src_res_np[:n_queries], transform),
                                   dtype=torch.float32, device=device)
    _, h, w = sim.shape
    cx, cy = cell_centers(w, size, device), cell_centers(h, size, device)
    sigma_px = float(cfg["constellation_sigma_px"])
    dx = cx[None, None, :] - expected_res[:, 0, None, None]
    dy = cy[None, :, None] - expected_res[:, 1, None, None]
    gauss_bonus = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma_px ** 2))  # [M, H, W] in (0, 1]

    refined_sim = sim[:n_queries] + prior_weight * gauss_bonus
    refined_res = locate(refined_sim, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
    out = to_original(refined_res.cpu(), trg_wh, size, letterbox=lb).numpy()
    return out, info
