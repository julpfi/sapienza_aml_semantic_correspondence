"""Keypoint transfer: source keypoint -> feature -> similarity map over target -> predicted location."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .backbone import preprocess


def sample_features(feat: torch.Tensor, kps_resized: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinearly sample feat [C,h,w] at keypoints [K,2] given in resized-pixel coords. -> [K,C]"""
    grid = (kps_resized / size) * 2 - 1                       # align_corners=False convention
    out = F.grid_sample(feat[None], grid[None, None], mode="bilinear", align_corners=False)
    return F.normalize(out[0, :, 0].T, dim=-1)


def cell_centers(n: int, size: int, device) -> torch.Tensor:
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * (size / n)


def locate(sim: torch.Tensor, size: int, matcher: str, window: int, tau: float) -> torch.Tensor:
    """sim [K,H,W] -> predicted (x, y) [K,2] in resized-pixel coords."""
    K, H, W = sim.shape
    cx, cy = cell_centers(W, size, sim.device), cell_centers(H, size, sim.device)
    flat = sim.reshape(K, -1).argmax(-1)
    r, c = flat // W, flat % W
    if matcher == "argmax":
        return torch.stack([cx[c], cy[r]], -1)
    # windowed soft-argmax: robust sub-patch refinement without being pulled by distant look-alikes
    half = window // 2
    offs = torch.arange(-half, half + 1, device=sim.device)
    rr = (r[:, None] + offs[None]).clamp(0, H - 1)              # [K,w]
    cc = (c[:, None] + offs[None]).clamp(0, W - 1)
    patch = sim[torch.arange(K)[:, None, None], rr[:, :, None], cc[:, None, :]]   # [K,w,w]
    wts = torch.softmax(patch.reshape(K, -1) / tau, -1).reshape(K, len(offs), len(offs))
    x = (wts.sum(1) * cx[cc]).sum(-1)
    y = (wts.sum(2) * cy[rr]).sum(-1)
    return torch.stack([x, y], -1)


class FeatureCache:
    """Small LRU cache of per-image features. Query-style test files repeat the same image pair
    for every keypoint, so this avoids recomputing the backbone. Create a new one per evaluation
    (never reuse across fine-tuning steps: the weights change)."""

    def __init__(self, max_items: int = 256):
        from collections import OrderedDict
        self.d, self.max = OrderedDict(), max_items
        self.hits = self.misses = 0

    def get(self, key):
        if key in self.d:
            self.d.move_to_end(key); self.hits += 1
            return self.d[key]
        self.misses += 1
        return None

    def put(self, key, val):
        self.d[key] = val
        if len(self.d) > self.max:
            self.d.popitem(last=False)


@torch.no_grad()
def extract(model, img, cfg, device, flip=False, cache=None, key=None) -> torch.Tensor:
    if cache is not None and key is not None:
        hit = cache.get((key, flip))
        if hit is not None:
            return hit
    f = model(preprocess(img, cfg.img_size, flip).to(device))[0]   # [C,h,w]
    if cache is not None and key is not None:
        cache.put((key, flip), f)
    return f


def alignment_score(fs: torch.Tensor, ft: torch.Tensor) -> float:
    """How well two feature maps align globally: mean best-match similarity of source patches."""
    a = fs.flatten(1).T
    b = ft.flatten(1)
    return (a @ b).max(-1).values.mean().item()


def predict_from_features(fs, ft, src_kps, src_wh, trg_wh, cfg, flipped=False) -> np.ndarray:
    """fs/ft [C,h,w]; src_kps [K,2] original coords -> predicted target kps [K,2] original coords."""
    S, dev = cfg.img_size, fs.device
    k = torch.as_tensor(src_kps, dtype=torch.float32, device=dev).clone()
    k[:, 0] *= S / src_wh[0]
    k[:, 1] *= S / src_wh[1]
    if flipped:
        k[:, 0] = S - k[:, 0]
    q = sample_features(fs, k, S)                                  # [K,C]
    if cfg.feat_upsample > 1:
        ft = F.interpolate(ft[None], scale_factor=cfg.feat_upsample, mode="bilinear", align_corners=False)[0]
        ft = F.normalize(ft, dim=0)
    sim = torch.einsum("kc,chw->khw", q, ft)
    xy = locate(sim, S, cfg.matcher, cfg.soft_window, cfg.soft_temperature)
    xy[:, 0] *= trg_wh[0] / S
    xy[:, 1] *= trg_wh[1] / S
    return xy.cpu().numpy()


@torch.no_grad()
def predict_pair(model, sample, cfg, device, cache=None) -> tuple[np.ndarray, bool]:
    """Returns (pred_trg_kps [K,2] in original target pixels, used_flip)."""
    fs = extract(model, sample["src_img"], cfg, device, cache=cache, key=sample.get("src_path"))
    ft = extract(model, sample["trg_img"], cfg, device, cache=cache, key=sample.get("trg_path"))
    flipped = False
    if cfg.flip_test:
        fs_f = extract(model, sample["src_img"], cfg, device, flip=True, cache=cache, key=sample.get("src_path"))
        if alignment_score(fs_f, ft) > alignment_score(fs, ft):
            fs, flipped = fs_f, True
    pred = predict_from_features(fs, ft, sample["src_kps"], sample["src_img"].size,
                                 sample["trg_img"].size, cfg, flipped)
    return pred, flipped
