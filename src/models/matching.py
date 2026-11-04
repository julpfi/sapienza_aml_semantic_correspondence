"""Keypoint transfer (Lang pipeline, unchanged logic):
source point -> its feature -> cosine similarity with every target cell -> predicted location.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from data.transforms import to_original, to_resized


def sample_point_features(feats, xy_resized, image_size):
    """Bilinearly sample feats [C, h, w] at points [K, 2] given in resized pixels -> [K, C]."""
    grid = (xy_resized / image_size) * 2 - 1  # align_corners=False convention
    out = F.grid_sample(feats[None], grid[None, None], mode="bilinear", align_corners=False)
    return F.normalize(out[0, :, 0].T, dim=-1)


def cell_centers(n_cells, image_size, device):
    return (torch.arange(n_cells, device=device, dtype=torch.float32) + 0.5) * (image_size / n_cells)


def locate(sim, image_size, matcher, window, temperature):
    """Similarity maps [K, H, W] -> predicted (x, y) [K, 2] in resized pixels."""
    n, h, w = sim.shape
    cx, cy = cell_centers(w, image_size, sim.device), cell_centers(h, image_size, sim.device)
    flat = sim.reshape(n, -1).argmax(-1)
    row, col = flat // w, flat % w
    if matcher == "argmax":
        return torch.stack([cx[col], cy[row]], -1)
    if matcher != "soft_window":
        raise ValueError(f"matcher must be 'argmax' or 'soft_window', got {matcher!r}")
    # Windowed soft-argmax: sub-cell refinement that is not pulled by distant look-alikes.
    half = window // 2
    offsets = torch.arange(-half, half + 1, device=sim.device)
    rows = (row[:, None] + offsets[None]).clamp(0, h - 1)  # [K, win]
    cols = (col[:, None] + offsets[None]).clamp(0, w - 1)
    patch = sim[torch.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]  # [K, win, win]
    weights = torch.softmax(patch.reshape(n, -1) / temperature, -1).reshape(n, len(offsets), len(offsets))
    x = (weights.sum(1) * cx[cols]).sum(-1)
    y = (weights.sum(2) * cy[rows]).sum(-1)
    return torch.stack([x, y], -1)


def alignment_score(src_feats, trg_feats):
    """How well two feature maps align globally: mean best-match similarity of source cells."""
    a = src_feats.flatten(1).T
    b = trg_feats.flatten(1)
    return (a @ b).max(-1).values.mean().item()


def predict_from_features(src_feats, trg_feats, src_xy, src_wh, trg_wh, cfg, flipped=False):
    """src_xy [K, 2] in original source pixels -> predicted [K, 2] in original target pixels."""
    size = cfg["image_size"]
    xy = to_resized(src_xy, src_wh, size).to(src_feats.device)
    if flipped:
        xy[:, 0] = size - xy[:, 0]
    query = sample_point_features(src_feats, xy, size)  # [K, C]
    if cfg["feat_upsample"] > 1:
        trg_feats = F.interpolate(trg_feats[None], scale_factor=cfg["feat_upsample"],
                                  mode="bilinear", align_corners=False)[0]
        trg_feats = F.normalize(trg_feats, dim=0)
    sim = torch.einsum("kc,chw->khw", query, trg_feats)
    pred = locate(sim, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
    return to_original(pred.cpu(), trg_wh, size).numpy()


class FeatureCache:
    """Small in-memory LRU cache of per-image features. Test files repeat images across
    queries, so this avoids recomputing the backbone. Never reuse one across training steps."""

    def __init__(self, max_items):
        self.items, self.max_items = OrderedDict(), max_items
        self.hits = self.misses = 0

    def get(self, key):
        if key in self.items:
            self.items.move_to_end(key)
            self.hits += 1
            return self.items[key]
        self.misses += 1
        return None

    def put(self, key, value):
        self.items[key] = value
        if len(self.items) > self.max_items:
            self.items.popitem(last=False)
