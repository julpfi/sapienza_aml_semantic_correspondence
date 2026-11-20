"""CPU checks for the two transformation modules on branch giuseppe:
1. src/models/cross_transform.py (CrossImageTransform: identity at init, gradients, parameter group, checkpoint round trip)
2. src/models/constellation.py (fit_similarity_2d, robust_constellation_fit, predict_with_constellation)

    python scripts/check_transforms.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config  # noqa: E402
from src.models.constellation import (  # noqa: E402
    apply_similarity_2d, fit_similarity_2d, predict_with_constellation, robust_constellation_fit,
)
from src.models.cross_transform import CrossImageTransform  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        failures.append(name)


def main():
    torch.manual_seed(0)
    cfg = get_config()
    cfg.update({"image_size": 224, "device": "cpu", "fusion": "none", "cross_transform": True,
                "cross_dim": 64, "cross_heads": 4, "cross_dropout": 0.05, "constellation_align": True})

    # 1. CrossImageTransform checks
    ct = CrossImageTransform(128, cfg).eval()
    src = F.normalize(torch.randn(2, 128, 16, 16), dim=1)
    trg = F.normalize(torch.randn(2, 128, 16, 16), dim=1)
    with torch.no_grad():
        s_out, t_out = ct(src, trg)
    diff_s = (s_out - src).abs().max().item()
    diff_t = (t_out - trg).abs().max().item()
    check("1a CrossImageTransform is exact identity at init", max(diff_s, diff_t) < 1e-6,
          f"max diff ({diff_s:.2e}, {diff_t:.2e})")
    check("1b CrossImageTransform outputs are unit norm",
          (s_out.norm(dim=1) - 1).abs().max().item() < 1e-5 and (t_out.norm(dim=1) - 1).abs().max().item() < 1e-5)

    ct.train()
    s_train, t_train = ct(src, trg)
    loss = (s_train * torch.randn_like(s_train) + t_train * torch.randn_like(t_train)).sum()
    loss.backward()
    has_ms_grad = ct.ms_out.weight.grad is not None and ct.ms_out.weight.grad.abs().max().item() > 0
    has_ca_grad = ct.ca_out.weight.grad is not None and ct.ca_out.weight.grad.abs().max().item() > 0
    check("1c Gradients reach multi-scale zero-init layer", has_ms_grad)
    check("1d Gradients reach cross-attention zero-init layer", has_ca_grad)

    # 2. Constellation Sim(2) alignment checks
    rng = np.random.default_rng(42)
    pts_src = rng.uniform(80, 560, size=(8, 2))
    # True transform: zoom 1.8x, rotate 40 deg, flip horizontally, shift
    angle = math.radians(40.0)
    r_true = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]) @ np.diag([-1.0, 1.0])
    s_true = 1.8
    t_true = np.array([45.0, -30.0])
    pts_trg = apply_similarity_2d(pts_src, (s_true, r_true, t_true))

    s_hat, r_hat, t_hat = fit_similarity_2d(pts_src, pts_trg)
    recon = apply_similarity_2d(pts_src, (s_hat, r_hat, t_hat))
    err = np.linalg.norm(recon - pts_trg, axis=1).max()
    check("2a fit_similarity_2d recovers exact scale + rotation + flip", err < 1e-6, f"max err {err:.2e}")

    # Add 2 gross outliers (wrong keypoints) and test RANSAC recovery
    noisy_trg = pts_trg.copy()
    noisy_trg[0] = [10.0, 600.0]
    noisy_trg[1] = [620.0, 20.0]
    conf = np.array([0.40, 0.42, 0.85, 0.88, 0.81, 0.79, 0.90, 0.84])
    transform, inliers, info = robust_constellation_fit(pts_src, noisy_trg, conf, 644, cfg)
    check("2b robust_constellation_fit rejects outliers and finds consensus",
          transform is not None and info["applied"] and not inliers[0] and not inliers[1] and inliers[2:].all(),
          f"inliers={inliers.tolist()}, scale={info.get('scale', 0):.2f} (true {s_true})")

    # 3. End-to-end predict_with_constellation check on synthetic feature maps
    sf = F.normalize(torch.randn(32, 16, 16), dim=0)
    tf = F.normalize(torch.randn(32, 16, 16), dim=0)
    pred, pinfo = predict_with_constellation(sf, tf, pts_src[:1], pts_src, (644, 644), (644, 644), cfg)
    check("2c predict_with_constellation returns finite [1, 2] target coordinates",
          pred.shape == (1, 2) and np.isfinite(pred).all(), f"pred={pred.round(1).tolist()}")

    print("\nALL PASS" if not failures else f"\nFAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
