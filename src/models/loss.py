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


def embedding_variance(feats):
    """1 - ||mean||^2 of L2-normalised rows [N, C]: 0 when all are equal (collapse), near 1 when spread out."""
    return 1 - F.normalize(feats.detach().float(), dim=-1).mean(0).pow(2).sum()


def peak_mass(probs, h, w, window):
    """Probability inside a window x window block of cells around the most likely cell.
    Near 1 for one peak, lower when the mass sits on several separate peaks."""
    probs = probs.reshape(h, w)
    row, col = divmod(int(probs.argmax()), w)
    half = window // 2
    return probs[max(row - half, 0):row + half + 1, max(col - half, 0):col + half + 1].sum()


def matching_loss(src_feats, trg_feats, src_xy_resized, trg_xys_resized, image_size, temperature, sigma,
                  peak_window=5):
    """src_feats, trg_feats [C, h, w]; src_xy_resized [1, 2]; trg_xys_resized [N, 2].

    Returns (loss, stats). stats (detached): `entropy` of the predicted distribution over the cells
    (falls when the matching gets sharper), `peak_mass` (peak_mass()) and `query`, the source descriptor [C].
    """
    _, h, w = trg_feats.shape
    query = sample_point_features(src_feats, src_xy_resized.to(src_feats.device), image_size)  # [1, C]
    logits = (query @ trg_feats.flatten(1)) / temperature  # [1, h*w]
    target = target_distribution(trg_xys_resized, h, w, image_size, sigma, trg_feats.device)
    log_probs = F.log_softmax(logits, -1)
    probs = log_probs.exp().detach()
    stats = {"entropy": -(probs * log_probs.detach()).sum(),
             "peak_mass": peak_mass(probs[0], h, w, peak_window),
             "query": query[0].detach()}
    return -(target[None] * log_probs).sum(-1).mean(), stats
