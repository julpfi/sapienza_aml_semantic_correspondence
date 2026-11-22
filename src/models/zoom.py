"""Two-stage object-centric zoom at inference (zero trainable parameters).

Enabled with cfg["object_zoom"] = True (off when False).
When an object is small in the target (or source) frame, matching at full-image 644x644 wastes
most of the 46x46 patch grid on background. Using the pair's `source_xy` context points from
`*_pairs.json`:
1. Stage 1 (coarse): match all `source_xy` context points on the full image -> coarse target pts.
2. Crop around the 5th-95th percentile bounding box (+ zoom_margin) on both source and target,
   skipping cropping on a side if the box already covers > zoom_max_area of the image.
3. Stage 2 (fine): re-extract features on the zoomed crops at full `image_size`, fuse Stage-1
   full-image similarity onto the Stage-2 crop grid (zoom_stage1_weight) to prevent crop-induced
   regressions, match the query, and map coordinates back to original target pixels.
"""

import numpy as np
import torch
import torch.nn.functional as F

from data.datasets import load_pair_source_xy
from data.transforms import to_original, to_resized
from src.models.matching import cell_centers, choose_source, locate, predict_from_features, sample_point_features


def bbox_crop(img, pts, margin=0.25, min_frac=0.20, pct_low=5.0, pct_high=95.0, max_area=1.0):
    """Crop `img` around the percentile bounding box of `pts` [K, 2] with a relative margin.
    Returns (cropped_pil_img, (x0, y0, x1, y1)); keeps full image if box covers > max_area."""
    w, h = img.size
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return img, (0, 0, w, h)
    lo = np.percentile(pts, pct_low, axis=0)
    hi = np.percentile(pts, pct_high, axis=0)
    span = np.maximum(hi - lo, np.array([w, h], dtype=np.float64) * float(min_frac))
    center = (lo + hi) * 0.5
    half = span * (0.5 + float(margin))
    x0, y0 = np.maximum(center - half, 0.0)
    x1, y1 = np.minimum(center + half, [float(w), float(h)])
    box = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return img, (0, 0, w, h)
    if ((box[2] - box[0]) * (box[3] - box[1])) / max(float(w * h), 1.0) > float(max_area):
        return img, (0, 0, w, h)
    return img.crop(box), box


def _upsample_trg(trg_feats, cfg):
    if cfg["feat_upsample"] > 1:
        trg_feats = F.interpolate(trg_feats[None], scale_factor=cfg["feat_upsample"],
                                  mode="bilinear", align_corners=False)[0]
        trg_feats = F.normalize(trg_feats, dim=0)
    return trg_feats


def _similarity_map(src_feats, trg_feats, src_xy, src_wh, cfg, flipped=False):
    """Compute [1, H, W] cosine similarity map for one source point [1, 2] in original pixels."""
    size = cfg["image_size"]
    xy = to_resized(src_xy, src_wh, size).to(src_feats.device)
    if flipped:
        xy[:, 0] = size - xy[:, 0]
    query = sample_point_features(src_feats, xy, size)  # [1, C]
    up_trg = _upsample_trg(trg_feats, cfg)
    return torch.einsum("kc,chw->khw", query, up_trg)


def _sample_stage1_into_crop(sim1, tbox, trg_wh, crop_hw, size):
    """Bilinearly sample Stage-1 full-image similarity [1, H1, W1] onto Stage-2 crop grid [1, H2, W2]."""
    h2, w2 = crop_hw[0], crop_hw[1]
    tc_wh = (tbox[2] - tbox[0], tbox[3] - tbox[1])
    cx_crop = cell_centers(w2, size, sim1.device)
    cy_crop = cell_centers(h2, size, sim1.device)
    x_full = cx_crop * (tc_wh[0] / size) + float(tbox[0])
    y_full = cy_crop * (tc_wh[1] / size) + float(tbox[1])
    gx = (x_full / float(trg_wh[0])) * 2.0 - 1.0
    gy = (y_full / float(trg_wh[1])) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)[None]  # [1, H2, W2, 2]
    return F.grid_sample(sim1[None], grid, mode="bilinear", align_corners=False)[0]


@torch.no_grad()
def _stage1_pair(model, sample, cfg, cache):
    """Run (or reuse) Stage 1 full-image features and flip decision for a pair."""
    src_feats = model.image_features(sample["src_img"], cache=cache, key=sample["src_image"])
    trg_feats = model.image_features(sample["trg_img"], cache=cache, key=sample["trg_image"])
    src_feats, trg_raw = model.pair_features(src_feats, trg_feats)
    src_flip, trg_flip = None, trg_raw
    if cfg["flip_test"]:
        src_flip_base = model.image_features(sample["src_img"], flip=True, cache=cache, key=sample["src_image"])
        src_flip, trg_flip = model.pair_features(src_flip_base, trg_feats)
    src_chosen, flipped = choose_source(src_feats, src_flip, trg_raw)
    trg_chosen = trg_flip if flipped else trg_raw
    return src_chosen, trg_chosen, flipped


@torch.no_grad()
def predict_with_zoom(model, sample, cfg, cache=None):
    """Two-stage object-centric zoom prediction for one query sample.
    Returns (pred_xy [2] in original target pixels, used_flip)."""
    src_chosen, trg_chosen, flipped = _stage1_pair(model, sample, cfg, cache)
    src_wh, trg_wh = sample["src_img"].size, sample["trg_img"].size
    size = int(cfg["image_size"])
    pair_key = ("zoom_pair", str(sample["pair_id"]), sample["src_image"], sample["trg_image"],
                size, bool(flipped))

    cached = cache.get(pair_key) if cache is not None else None
    if cached is not None:
        sbox, tbox, fs2, ft2, ctx_xy = cached
    else:
        ctx_xy = load_pair_source_xy(cfg, sample["pair_id"], fallback_xy=sample["src_xy"][None])
        coarse_ctx = predict_from_features(src_chosen, trg_chosen, ctx_xy, src_wh, trg_wh, cfg, flipped)
        max_area = float(cfg.get("zoom_max_area", 1.0))
        s_crop, sbox = bbox_crop(sample["src_img"], ctx_xy, cfg["zoom_margin"], cfg["zoom_min_frac"],
                                 cfg["zoom_pct_low"], cfg["zoom_pct_high"], max_area=max_area)
        t_crop, tbox = bbox_crop(sample["trg_img"], coarse_ctx, cfg["zoom_margin"], cfg["zoom_min_frac"],
                                 cfg["zoom_pct_low"], cfg["zoom_pct_high"], max_area=max_area)
        full_src = sbox == (0, 0, src_wh[0], src_wh[1])
        full_trg = tbox == (0, 0, trg_wh[0], trg_wh[1])
        if full_src and full_trg:
            fs2, ft2 = src_chosen, trg_chosen
        else:
            fs2 = src_chosen if full_src else model.image_features(
                s_crop, flip=flipped, cache=cache, key=("zoom_src", str(sample["pair_id"]), sbox, size))
            ft2 = trg_chosen if full_trg else model.image_features(
                t_crop, cache=cache, key=("zoom_trg", str(sample["pair_id"]), tbox, size))
            if not (full_src and full_trg):
                fs2, ft2 = model.pair_features(fs2, ft2)
        if cache is not None:
            cache.put(pair_key, (sbox, tbox, fs2, ft2, ctx_xy))

    sc_wh = (sbox[2] - sbox[0], sbox[3] - sbox[1])
    tc_wh = (tbox[2] - tbox[0], tbox[3] - tbox[1])
    q_in = np.asarray(sample["src_xy"], dtype=np.float32) - np.array(sbox[:2], dtype=np.float32)
    inside = (0.0 <= q_in[0] < sc_wh[0]) and (0.0 <= q_in[1] < sc_wh[1])
    if not inside:
        pred = predict_from_features(src_chosen, trg_chosen, sample["src_xy"][None], src_wh, trg_wh, cfg, flipped)
        return pred[0], flipped

    sim2 = _similarity_map(fs2, ft2, q_in[None], sc_wh, cfg, flipped)
    w1 = float(cfg.get("zoom_stage1_weight", 0.0))
    is_full = (sbox == (0, 0, src_wh[0], src_wh[1])) and (tbox == (0, 0, trg_wh[0], trg_wh[1]))
    if not is_full and w1 > 0.0:
        sim1 = _similarity_map(src_chosen, trg_chosen, sample["src_xy"][None], src_wh, cfg, flipped)
        if float(sim1.max()) > float(sim2.max()) + 0.05:
            pred1 = locate(sim1, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
            return to_original(pred1.cpu(), trg_wh, size).numpy()[0], flipped
        sim1_crop = _sample_stage1_into_crop(sim1, tbox, trg_wh, sim2.shape[1:], size)
        sim2 = (1.0 - w1) * sim2 + w1 * sim1_crop

    pred2 = locate(sim2, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
    pred_xy = to_original(pred2.cpu(), tc_wh, size).numpy()[0]
    return pred_xy + np.array(tbox[:2], dtype=np.float32), flipped
