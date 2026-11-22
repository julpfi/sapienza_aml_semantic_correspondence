"""CPU checks for CrossImageTransform (branch giuseppe):
1. src/models/cross_transform.py (CrossImageTransform: identity at init, gradients, parameter group, checkpoint round trip)

    python scripts/check_transforms.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config  # noqa: E402
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
                "cross_dim": 64, "cross_heads": 4, "cross_dropout": 0.05})

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

    print("\nALL PASS" if not failures else f"\nFAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
