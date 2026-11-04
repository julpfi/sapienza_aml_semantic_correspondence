"""Soft-label matching loss (Lang finetune.py), extended to several accepted targets.

For a source point, its feature is compared with every target cell. A softmax over the cells
is trained to put its mass on the accepted targets: a Gaussian bump (sigma in cells) at each
accepted target, summed and normalised. With one target this is exactly the Lang loss; with
several, any accepted location (for example any wheel) counts as correct.
"""

import torch
import torch.nn.functional as F

from src.models.matching import cell_centers, sample_point_features


def target_distribution(trg_xys_resized, h, w, image_size, sigma, device):
    """Accepted targets [N, 2] (resized pixels) -> probability over the h*w cells."""
    cell = image_size / w
    trg = trg_xys_resized.to(device)
    if sigma > 0:
        cx, cy = cell_centers(w, image_size, device), cell_centers(h, image_size, device)
        dx = (cx[None, None, :] - trg[:, 0, None, None]) / cell
        dy = (cy[None, :, None] - trg[:, 1, None, None]) / cell
        bumps = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)).flatten(1)  # [N, h*w]
        bumps = bumps / bumps.sum(-1, keepdim=True)  # each target weighs the same
        target = bumps.sum(0)
    else:
        col = (trg[:, 0] / cell).long().clamp(0, w - 1)
        row = (trg[:, 1] / cell).long().clamp(0, h - 1)
        target = F.one_hot(row * w + col, h * w).float().sum(0)
    return target / target.sum()


def matching_loss(src_feats, trg_feats, src_xy_resized, trg_xys_resized, image_size, temperature, sigma):
    """src_feats, trg_feats [C, h, w]; src_xy_resized [1, 2]; trg_xys_resized [N, 2].

    Returns (loss, entropy): entropy of the predicted distribution over the cells (detached), which
    falls when the matching gets sharper.
    """
    _, h, w = trg_feats.shape
    query = sample_point_features(src_feats, src_xy_resized.to(src_feats.device), image_size)  # [1, C]
    logits = (query @ trg_feats.flatten(1)) / temperature  # [1, h*w]
    target = target_distribution(trg_xys_resized, h, w, image_size, sigma, trg_feats.device)
    log_probs = F.log_softmax(logits, -1)
    entropy = -(log_probs.exp() * log_probs).sum(-1).mean().detach()
    return -(target[None] * log_probs).sum(-1).mean(), entropy
