"""Matching losses with several accepted targets. Pick one with cfg["loss"].

For a source point, its feature is compared with every target cell; a softmax over the cells
(temperature loss_temperature) gives a probability p for each cell. Two ways to train p:

"soft_ce" (v2 default): cross-entropy with a target distribution made of a Gaussian bump
    (label_sigma, in cells) on each accepted target, every target with the same weight. The model
    must spread p over the whole bump and over ALL accepted targets (e.g. every wheel), so the loss
    has a floor above 0 and the entropy of p stays high (about 3.3-3.5 nats with sigma = 1).
"marginal" (Lang pipeline): -log of the total p on the positive cells, the cells whose centre is
    within marginal_radius cells of ANY accepted target (plus the nearest cell of each target).
    How p is shared among the positives is free, so the model can put almost all of it on one
    cell of one target: lowest loss 0, low entropy, and it commits to a single target, which is
    what PCK rewards (a point between two wheels is wrong for both). label_sigma is not used.
"""

import torch
import torch.nn.functional as F

from src.models.matching import cell_centers, sample_point_features

LOSSES = ("soft_ce", "marginal")


def _cell_distances(trg_xys_resized, h, w, image_size, device):
    """Distance in cells from each accepted target [N, 2] (resized pixels) to each cell centre -> [N, h*w]."""
    cell = image_size / w
    trg = trg_xys_resized.to(device)
    cx, cy = cell_centers(w, image_size, device), cell_centers(h, image_size, device)
    dx = (cx[None, None, :] - trg[:, 0, None, None]) / cell
    dy = (cy[None, :, None] - trg[:, 1, None, None]) / cell
    return (dx ** 2 + dy ** 2).sqrt().flatten(1)


def target_distribution(trg_xys_resized, h, w, image_size, sigma, device):
    """soft_ce target: accepted targets [N, 2] (resized pixels) -> probability over the h*w cells."""
    if sigma > 0:
        dist = _cell_distances(trg_xys_resized, h, w, image_size, device)
        bumps = torch.exp(-dist ** 2 / (2 * sigma ** 2))  # [N, h*w]
        target = (bumps / bumps.sum(-1, keepdim=True)).sum(0)  # each target weighs the same
    else:
        cell = image_size / w
        trg = trg_xys_resized.to(device)
        col = (trg[:, 0] / cell).long().clamp(0, w - 1)
        row = (trg[:, 1] / cell).long().clamp(0, h - 1)
        target = F.one_hot(row * w + col, h * w).float().sum(0)
    return target / target.sum()


def positive_cells(trg_xys_resized, h, w, image_size, radius, device):
    """marginal positives: bool [h*w], cells within `radius` cells of any accepted target, plus the
    nearest cell of each target (so a target always has at least one positive)."""
    dist = _cell_distances(trg_xys_resized, h, w, image_size, device)
    mask = (dist <= radius).any(0)
    mask[dist.argmin(-1)] = True
    return mask


def matching_loss(src_feats, trg_feats, src_xy_resized, trg_xys_resized, cfg):
    """src_feats, trg_feats [C, h, w]; src_xy_resized [1, 2]; trg_xys_resized [N, 2] (resized pixels).

    Returns (loss, entropy): entropy (detached) of p over the cells, which falls when the matching
    gets sharper.
    """
    _, h, w = trg_feats.shape
    size = cfg["image_size"]
    query = sample_point_features(src_feats, src_xy_resized.to(src_feats.device), size)  # [1, C]
    log_probs = F.log_softmax((query @ trg_feats.flatten(1))[0] / cfg["loss_temperature"], -1)  # [h*w]
    entropy = -(log_probs.exp() * log_probs).sum().detach()
    if cfg["loss"] == "soft_ce":
        target = target_distribution(trg_xys_resized, h, w, size, cfg["label_sigma"], trg_feats.device)
        return -(target * log_probs).sum(), entropy
    if cfg["loss"] == "marginal":
        mask = positive_cells(trg_xys_resized, h, w, size, cfg["marginal_radius"], trg_feats.device)
        return -torch.logsumexp(log_probs[mask], 0), entropy
    raise ValueError(f"loss must be one of {LOSSES}, got {cfg['loss']!r}")
