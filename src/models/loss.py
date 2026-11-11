"""Matching losses with several accepted targets. Pick one with cfg["loss"].

For a source point, its feature is compared with every target cell; a softmax over the cells
(temperature loss_temperature) gives a probability p for each cell. Two ways to train p:

"soft_ce" (v2 default): cross-entropy with a target distribution made of a Gaussian bump
    (label_sigma, in cells) on each accepted target, every target with the same weight. The model
    must spread p over the whole bump and over ALL accepted targets (e.g. every wheel), so the loss
    has a floor above 0 and the entropy of p stays high (about 3.3-3.5 nats with sigma = 1, i.e.
    entropy_norm 0.43-0.46 on the 46 x 46 grid).
"marginal" (Lang pipeline): -log of the total p on the positive cells, the cells whose centre is
    within marginal_radius cells of ANY accepted target (plus the nearest cell of each target).
    How p is shared among the positives is free, so the model can put almost all of it on one
    cell of one target: lowest loss 0, low entropy, and it commits to a single target, which is
    what PCK rewards (a point between two wheels is wrong for both). label_sigma is not used.
"""

import math

import torch
import torch.nn.functional as F

from src.models.matching import cell_centers, sample_point_features

LOSSES = ("soft_ce", "marginal")
LOSS_STATS = ("loss", "loss_over_chance", "entropy_norm")  # what matching_loss returns, in this order
# Logged and reported (train/, val/): the normalised stats only. The raw loss depends on the loss type,
# the temperature and the grid size, so it is used for the gradients and never charted.
LOGGED_STATS = LOSS_STATS[1:]


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


def matching_loss(src_feats, trg_feats, src_xy_resized, trg_xys_resized, cfg, temperature=None):
    """src_feats, trg_feats [C, h, w]; src_xy_resized [1, 2]; trg_xys_resized [N, 2] (resized pixels).
    temperature: softmax temperature, default cfg["loss_temperature"] (train.py passes the annealed one).

    Returns the three LOSS_STATS; only `loss` carries gradients:
        loss              the training loss, in nats
        loss_over_chance  (loss - floor) / (chance - floor): 1 = uniform p (chance), 0 = lowest possible loss.
                          chance = log(n_cells) for soft_ce and log(n_cells / n_positive) for marginal;
                          floor = entropy of the soft_ce target, 0 for marginal.
        entropy_norm      H(p) / log(n_cells): 0 = one-hot, 1 = uniform over the cells
    """
    _, h, w = trg_feats.shape
    size, log_cells = cfg["image_size"], math.log(h * w)
    query = sample_point_features(src_feats, src_xy_resized.to(src_feats.device), size)  # [1, C]
    logits = (query @ trg_feats.flatten(1))[0] / (temperature or cfg["loss_temperature"])
    log_probs = F.log_softmax(logits, -1)  # [h*w]
    entropy_norm = -(log_probs.exp() * log_probs).sum().detach() / log_cells
    if cfg["loss"] == "soft_ce":
        target = target_distribution(trg_xys_resized, h, w, size, cfg["label_sigma"], trg_feats.device)
        loss = -(target * log_probs).sum()
        chance, floor = log_cells, float(-torch.xlogy(target, target).sum())
    elif cfg["loss"] == "marginal":
        mask = positive_cells(trg_xys_resized, h, w, size, cfg["marginal_radius"], trg_feats.device)
        loss = -torch.logsumexp(log_probs[mask], 0)
        chance, floor = log_cells - math.log(int(mask.sum())), 0.0
    else:
        raise ValueError(f"loss must be one of {LOSSES}, got {cfg['loss']!r}")
    return loss, (loss.detach() - floor) / (chance - floor), entropy_norm
