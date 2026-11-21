"""Two-stage object-centric zoom at inference (zero trainable parameters).

Enabled with cfg["object_zoom"] = True (off when False).
When an object is small in the target (or source) frame, matching at full-image 644x644 wastes
most of the 46x46 patch grid on background. Using the pair's `source_xy` context points from
`*_pairs.json`:
1. Stage 1 (coarse): match all `source_xy` context points on the full image -> coarse target pts.
2. Crop around the 5th-95th percentile bounding box (+ zoom_margin) on both source and target.
3. Stage 2 (fine): re-extract features on the zoomed crops at full `image_size`, match the query
   (optionally with `constellation_align`), and map coordinates back to original target pixels.
"""

import numpy as np
import torch

from src.models.constellation import load_pair_source_xy, predict_with_constellation
from src.models.matching import choose_source, predict_from_features


def bbox_crop(img, pts, margin=0.25, min_frac=0.20, pct_low=5.0, pct_high=95.0):
    """Crop `img` around the percentile bounding box of `pts` [K, 2] with a relative margin.
    Returns (cropped_pil_img, (x0, y0, x1, y1))."""
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
    return img.crop(box), box


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
    pair_key = ("zoom_pair", str(sample["pair_id"]), sample["src_image"], sample["trg_image"],
                int(cfg["image_size"]), bool(flipped))

    cached = cache.get(pair_key) if cache is not None else None
    if cached is not None:
        sbox, tbox, fs2, ft2, ctx_xy = cached
    else:
        ctx_xy = load_pair_source_xy(cfg, sample["pair_id"], fallback_xy=sample["src_xy"][None])
        coarse_ctx = predict_from_features(src_chosen, trg_chosen, ctx_xy, src_wh, trg_wh, cfg, flipped)
        s_crop, sbox = bbox_crop(sample["src_img"], ctx_xy, cfg["zoom_margin"], cfg["zoom_min_frac"],
                                 cfg["zoom_pct_low"], cfg["zoom_pct_high"])
        t_crop, tbox = bbox_crop(sample["trg_img"], coarse_ctx, cfg["zoom_margin"], cfg["zoom_min_frac"],
                                 cfg["zoom_pct_low"], cfg["zoom_pct_high"])
        fs2 = model.image_features(s_crop, flip=flipped, cache=cache,
                                   key=("zoom_src", str(sample["pair_id"]), sbox, int(cfg["image_size"])))
        ft2 = model.image_features(t_crop, cache=cache,
                                   key=("zoom_trg", str(sample["pair_id"]), tbox, int(cfg["image_size"])))
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

    if cfg.get("constellation_align"):
        ctx_in = np.asarray(ctx_xy, dtype=np.float32) - np.array(sbox[:2], dtype=np.float32)
        keep = (ctx_in[:, 0] >= 0.0) & (ctx_in[:, 0] < sc_wh[0]) & (ctx_in[:, 1] >= 0.0) & (ctx_in[:, 1] < sc_wh[1])
        ctx_valid = ctx_in[keep] if keep.any() else q_in[None]
        pred = predict_with_constellation(fs2, ft2, q_in[None], ctx_valid, sc_wh, tc_wh, cfg, flipped)[0][0]
    else:
        pred = predict_from_features(fs2, ft2, q_in[None], sc_wh, tc_wh, cfg, flipped)[0]

    return pred + np.array(tbox[:2], dtype=np.float32), flipped
