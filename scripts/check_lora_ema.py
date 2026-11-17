"""Check LoRA (identity at init, trainable set, gradients, checkpoint round trip) and EMA, on CPU.

    python scripts/check_lora_ema.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.backbone import DinoBackbone  # noqa: E402
from src.models.lora import LoRALinear, apply_lora, lora_parameters  # noqa: E402
from src.utils.checkpoint import load_model_weights, trainable_state  # noqa: E402
from src.utils.ema import EMA  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        failures.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


def make_cfg():
    cfg = get_config()
    cfg.update({"image_size": 224, "device": "cpu", "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                "lora_blocks": 2, "fusion": "none"})
    return cfg


def wrap_once(model, cfg):
    """apply_lora on the backbone unless the model already did it."""
    if not lora_parameters(model.backbone):
        apply_lora(model.backbone, cfg)
    model.backbone.freeze()
    for p in lora_parameters(model):
        p.requires_grad_(True)
    return model.eval()


def randomise_B(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def main():
    torch.manual_seed(0)
    cfg = make_cfg()
    n_blocks = cfg["lora_blocks"]
    x = torch.randn(1, 3, 224, 224)

    # 1. identity at init
    bb = DinoBackbone(cfg).eval()
    with torch.no_grad():
        before = bb(x)
    params = apply_lora(bb, cfg)
    with torch.no_grad():
        after = bb(x)
    diff = (before - after).abs().max().item()
    check("1 identity at init", diff < 1e-6, f"max diff {diff:.2e}")

    # 2. trainable set
    bb.freeze()
    check("2a freeze() also freezes LoRA", not any(p.requires_grad for p in bb.parameters()))
    for p in lora_parameters(bb):
        p.requires_grad_(True)
    trainable = [n for n, p in bb.named_parameters() if p.requires_grad]
    expected = 2 * n_blocks * 2
    check("2b only LoRA parameters trainable", all("lora_" in n for n in trainable) and len(trainable) == expected,
          f"{len(trainable)} tensors (expected {expected}), {sum(p.numel() for p in lora_parameters(bb)):,} trainable parameters")
    check("2c apply_lora returned the LoRA parameters", len(params) == expected)
    bb.train()
    modes = [any(isinstance(m, LoRALinear) for m in b.modules()) == b.training for b in bb.blocks]
    check("2d only LoRA blocks are in train mode", all(modes) and not bb.model.embeddings.training)

    # 3. gradients
    randomise_B(bb)
    bb.eval()
    out = bb(x)
    check("3a features change with lora_B", (out - before).abs().max().item() > 1e-4)
    out.square().sum().backward()
    lora_ok = all(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_parameters(bb))
    base_ok = all(p.grad is None for n, p in bb.named_parameters() if "lora_" not in n)
    check("3b gradients reach lora_A / lora_B", lora_ok)
    check("3c no gradient on base weights", base_ok)

    # 4. checkpoint round trip
    m1 = wrap_once(build_model(cfg), cfg)
    randomise_B(m1)
    state = trainable_state(m1)
    check("4a saved state holds only lora keys", len(state) > 0 and all("lora_" in k for k in state),
          f"{len(state)} keys")
    m2 = wrap_once(build_model(cfg), cfg)
    load_model_weights(m2, state)
    with torch.no_grad():
        d = (m1(x) - m2(x)).abs().max().item()
    check("4b reload reproduces features", d < 1e-6, f"max diff {d:.2e}")
    check("4c keys stable across rebuild", set(trainable_state(m2)) == set(state))

    # 5. idempotence
    n_before = len(list(bb.parameters()))
    apply_lora(bb, cfg)
    wrapped_twice = any(isinstance(m.base, LoRALinear) for m in bb.modules() if isinstance(m, LoRALinear))
    check("5 apply_lora idempotent", len(list(bb.parameters())) == n_before and not wrapped_twice)

    # 6. EMA
    lin = nn.Linear(3, 2)
    frozen = nn.Linear(2, 2)
    box = nn.ModuleDict({"lin": lin, "frozen": frozen})
    for p in frozen.parameters():
        p.requires_grad_(False)
    ema = EMA(box, 0.9)
    check("6a tracks only trainable parameters", set(ema.state_dict()) == {"lin.weight", "lin.bias"})
    w0 = lin.weight.detach().clone()
    w1 = w0 + 1.0
    with torch.no_grad():
        lin.weight.copy_(w1)
    ema.update()
    want = 0.9 * w0 + 0.1 * w1
    check("6b update maths", torch.allclose(ema.state_dict()["lin.weight"], want, atol=1e-7))
    with torch.no_grad():
        lin.weight.copy_(w1 + 1.0)
    ema.update()
    want = 0.9 * want + 0.1 * (w1 + 1.0)
    check("6c second update maths", torch.allclose(ema.state_dict()["lin.weight"], want, atol=1e-6))
    live = lin.weight.detach().clone()
    ema.swap_in()
    check("6d swap_in loads shadow", torch.equal(lin.weight, ema.state_dict()["lin.weight"]))
    ema.swap_out()
    check("6e swap_out restores exactly", torch.equal(lin.weight, live))
    ema.swap_in()
    check("6f double swap_in raises", raises(ema.swap_in, RuntimeError))
    ema.swap_out()
    check("6g swap_out without swap_in raises", raises(ema.swap_out, RuntimeError))
    ema2 = EMA(box, 0.9)
    ema2.load_state_dict(ema.state_dict())
    check("6h state_dict round trip", all(torch.equal(ema.state_dict()[k], v) for k, v in ema2.state_dict().items()))
    bad = ema.state_dict()
    bad["nope"] = torch.zeros(1)
    check("6i unknown name raises", raises(lambda: ema2.load_state_dict(bad), KeyError))
    del bad["nope"], bad["lin.bias"]
    check("6j missing name raises", raises(lambda: ema2.load_state_dict(bad), KeyError))

    print("\nALL PASS" if not failures else f"\nFAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
