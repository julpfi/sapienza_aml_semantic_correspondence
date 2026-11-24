"""Check the Stable Diffusion features and the DINOv2 + SD fusion on CPU, with a tiny SD test pipeline.

    python scripts/check_sd_fusion.py

Uses hf-internal-testing/tiny-stable-diffusion-torch (random weights, a few MB), never the real SD 2.1.
The fusion "none" checks run first, before anything imports diffusers.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import build_dataset  # noqa: E402
from src.config import get_config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.checkpoint import load_model_weights, trainable_state  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        failures.append(name)


def make_cfg(**overrides):
    cfg = get_config()
    # Tiny pipeline: VAE /2 and UNet /2, so up-block 1 is at 1/2 of sd_image_size (64 -> 32 cells),
    # bilinearly resized to the DINOv2 grid (224 / 14 = 16): the resize path is exercised too.
    cfg.update({"sd_model_id": "hf-internal-testing/tiny-stable-diffusion-torch",
                "sd_model_revision": "a88cdfbd91f96ec7f61eb7484b652ff0f4ee701d",
                "image_size": 224, "sd_image_size": 64, "device": "cpu", "max_train_queries": 1,
                "cross_transform": False, "lora_train_layernorm": False, "object_zoom": False})
    cfg.update(overrides)
    return cfg


def head_model_class():
    """CorrespondenceModel from the v2 (e5cedb8) commit, imported from a temp folder."""
    code = subprocess.run(["git", "show", "e5cedb8:src/models/__init__.py"], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "head_models.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location("head_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CorrespondenceModel


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    grid = (16, 16)

    # 6. fusion "none": no diffusers, same keys and features as v2 (e5cedb8).
    cfg_none = make_cfg(fusion="none")
    none_model = build_model(cfg_none)
    check("none: diffusers not imported", "diffusers" not in sys.modules)
    head_model = head_model_class()(cfg_none).eval()
    check("none: state-dict keys equal the v2 (e5cedb8) model's (git show e5cedb8:src/models/__init__.py)",
          list(none_model.state_dict()) == list(head_model.state_dict()))
    with torch.no_grad():
        none_feats = none_model(x)
        diff = (none_feats - head_model(x)).abs().max().item()
    check("none: features equal the v2 (e5cedb8) model's", diff == 0.0, f"(max abs diff {diff:.2e})")
    del head_model

    # 1. SDFeatures output.
    adapter_model = build_model(make_cfg(fusion="adapter"))
    sd = adapter_model.sd
    feats = sd(x, grid)
    check("sd: shape [B, out_dim, h, w]", tuple(feats.shape) == (2, sd.out_dim, *grid), f"{tuple(feats.shape)}")
    check("sd: float32", feats.dtype == torch.float32)
    norms = feats.norm(dim=1)
    check("sd: unit norm over channels", (norms - 1).abs().max().item() < 1e-5)
    check("sd: no NaN", not torch.isnan(feats).any().item())

    # 2. Determinism, frozen, eval.
    again = sd(x, grid)
    check("sd: same image twice -> identical features", torch.equal(feats, again))
    single = sd(x[1:], grid)
    check("sd: features do not depend on the batch", (single[0] - feats[1]).abs().max().item() < 1e-5)
    other = sd(torch.randn(1, 3, 224, 224), grid)
    check("sd: different image -> different features", (other[0] - feats[0]).abs().max().item() > 1e-3)
    check("sd: no parameter requires grad", not any(p.requires_grad for p in sd.parameters()))
    adapter_model.train()
    check("sd: model.train() leaves the SD module in eval mode",
          not any(m.training for m in sd.modules()) and adapter_model.fusion.training)
    adapter_model.eval()

    # 3. Adapter at init: fused == dino, the zero layer gets a gradient.
    from src.models.fusion import Fusion

    cfg = make_cfg()
    dino = torch.nn.functional.normalize(torch.randn(2, 768, *grid), dim=1)
    adapter = Fusion("adapter", 768, sd.out_dim, cfg)
    fused = adapter(dino, feats)
    diff = (fused - dino).abs().max().item()
    check("adapter: fused == dino at init", diff < 1e-6, f"(max abs diff {diff:.2e})")
    (fused * torch.randn_like(fused)).sum().backward()
    grad = adapter.out.weight.grad
    check("adapter: final zero layer has a non-zero gradient", grad is not None and grad.abs().max().item() > 0,
          f"(|grad| max {grad.abs().max().item():.2e})" if grad is not None else "")

    # 3b. adapter_mix: init equals the adapter output, and no dead branch.
    mix = Fusion("adapter_mix", 768, sd.out_dim, cfg).eval()
    adapter_ref = Fusion("adapter", 768, sd.out_dim, cfg).eval()
    check("adapter_mix: fused == normalize(dino) at init", (mix(dino, feats) - dino).abs().max().item() < 1e-6)
    with torch.no_grad():
        for p in mix.out.parameters():
            p.normal_(0, 0.05)
    adapter_ref.load_state_dict({k: v for k, v in mix.state_dict().items() if not k.startswith("mix.")})
    diff = (mix(dino, feats) - adapter_ref(dino, feats)).abs().max().item()
    check("adapter_mix: equals the adapter output at init (same inp/blocks/out)", diff < 1e-6, f"(max abs diff {diff:.2e})")
    check("adapter_mix: same adapter module names", set(adapter_ref.state_dict())
          == {k for k in mix.state_dict() if not k.startswith("mix.")})
    fresh_mix = Fusion("adapter_mix", 768, sd.out_dim, cfg)
    fresh_mix.train()
    toy_dino = torch.randn(2, 768, *grid)
    toy_sd = torch.randn(2, sd.out_dim, *grid)
    loss = (fresh_mix(toy_dino, toy_sd) * torch.randn(2, 768, *grid)).sum()
    loss.backward()
    check("adapter_mix: grad of out.weight non-zero at init", fresh_mix.out.weight.grad.abs().max().item() > 0)
    # At init s = 0, so the s half of mix has a zero gradient; after one step out != 0 and it must receive one.
    optimizer = torch.optim.SGD(fresh_mix.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    (fresh_mix(toy_dino, toy_sd) * torch.randn(2, 768, *grid)).sum().backward()
    w = fresh_mix.mix.weight.grad
    check("adapter_mix: grads of both halves of mix.weight non-zero",
          w[:, :768].abs().max().item() > 0 and w[:, 768:].abs().max().item() > 0)
    check("adapter_mix: out_dim = dino_dim", fresh_mix.out_dim == 768 and fresh_mix(toy_dino, toy_sd).shape[1] == 768)

    # 4. Concat.
    concat = Fusion("concat", 768, sd.out_dim, cfg)
    out = concat(dino, feats)
    check("concat: output dim = dino_dim + sd_dim", out.shape[1] == 768 + sd.out_dim, f"({out.shape[1]})")
    check("concat: unit norm", (out.norm(dim=1) - 1).abs().max().item() < 1e-5)
    check("concat: no parameters", not list(concat.parameters()))

    # 5. Full model.
    with torch.no_grad():
        diff = (adapter_model(x) - none_feats).abs().max().item()
    check("model adapter: features equal the fusion 'none' model's", diff < 1e-5, f"(max abs diff {diff:.2e})")
    concat_cfg = make_cfg(fusion="concat")
    concat_model = build_model(concat_cfg)
    sample = build_dataset(concat_cfg, "train")[0]
    for flip_test in (False, True):
        concat_cfg["flip_test"] = flip_test
        cache = concat_model.new_cache()
        pred, flipped = concat_model.predict(sample, concat_cfg, cache=cache)
        pred = np.asarray(pred, dtype=np.float64)
        w, h = sample["trg_img"].size
        inside = np.isfinite(pred).all() and 0 <= pred[0] <= w and 0 <= pred[1] <= h
        check(f"model concat: predict flip_test={flip_test} -> finite point inside the target image", inside,
              f"({pred.round(1).tolist()} in {w}x{h}, flipped {flipped})")
    hit = cache.get((sample["src_image"], True))
    check("model concat: flip_test caches the mirrored source under (key, True)",
          hit is not None and hit.shape[0] == 768 + concat_model.sd.out_dim)

    # 7. Checkpoint: only the adapter, and a round trip reproduces the features.
    with torch.no_grad():
        for p in adapter_model.fusion.parameters():
            p.add_(0.01 * torch.randn_like(p))
    adapter_model.fusion.requires_grad_(True)
    state = trainable_state(adapter_model)
    check("checkpoint: only fusion. keys", bool(state) and all(k.startswith("fusion.") for k in state),
          f"({len(state)} keys)")
    fresh = build_model(make_cfg(fusion="adapter"))
    load_model_weights(fresh, state)
    with torch.no_grad():
        diff = (fresh(x) - adapter_model(x)).abs().max().item()
        moved = (fresh(x) - none_feats).abs().max().item()
    check("checkpoint: reload reproduces the features", diff < 1e-6 and moved > 1e-4,
          f"(max abs diff {diff:.2e}; vs DINOv2 {moved:.2e})")

    # 8. D-009: multi-block features, noise ensemble, adapter dropout, train / eval sample count, preset.
    import torch.nn.functional as F

    multi_model = build_model(make_cfg(fusion="adapter", sd_up_blocks=[0, 1], sd_noise_samples=2,
                                       adapter_dropout=0.5))
    msd = multi_model.sd
    channels = list(reversed(msd.unet.config.block_out_channels))
    check("sd multi: out_dim = sum of block channels", msd.out_dim == channels[0] + channels[1], f"({msd.out_dim})")
    mfeats = msd(x, grid)
    check("sd multi: shape and unit norm", tuple(mfeats.shape) == (2, msd.out_dim, *grid)
          and (mfeats.norm(dim=1) - 1).abs().max().item() < 1e-5)
    # Rebuild the expected tensor from the single-block hooks: each block normalised, then the concat normalised.
    expected = []
    for block in (0, 1):
        single_model = build_model(make_cfg(fusion="adapter", sd_up_block=block))
        expected.append(single_model.sd(x, grid))
        del single_model
    manual = F.normalize(torch.cat(expected, dim=1), dim=1)
    diff = (manual - mfeats).abs().max().item()
    check("sd multi: equals the per-block normalised concat", diff < 1e-5, f"(max abs diff {diff:.2e})")
    sd1 = build_model(make_cfg(fusion="adapter", sd_up_block=1)).sd
    check("sd: sd_up_blocks None equals the single-block features", torch.equal(sd1(x, grid), sd1(x, grid, 1)))
    f1, f2 = msd(x, grid, 1), msd(x, grid, 2)
    check("sd: n_samples=2 deterministic", torch.equal(f2, msd(x, grid, 2)))
    check("sd: n_samples=2 differs from n_samples=1", (f1 - f2).abs().max().item() > 1e-6)
    check("sd: n_samples=1 equals the default call", torch.equal(f1, msd(x, grid)))

    drop_adapter = Fusion("adapter", 768, sd.out_dim, make_cfg(adapter_dropout=0.5)).eval()
    diff = (drop_adapter(dino, feats) - dino).abs().max().item()
    check("adapter dropout: eval mode at init gives fused == dino", diff < 1e-6, f"(max abs diff {diff:.2e})")

    calls = []
    real_forward = multi_model.sd.forward
    multi_model.sd.forward = lambda xx, hw, n=1: (calls.append(n), real_forward(xx, hw, n))[1]
    multi_model.train()
    multi_model(x[:1])
    multi_model.eval()
    multi_model(x[:1])
    check("model: train uses 1 noise sample, eval uses sd_noise_samples", calls == [1, 2], f"({calls})")

    preset = get_config("unified_lora_large_sd")
    check("preset unified_lora_large_sd keys", preset["fusion"] == "adapter" and preset["sd_up_blocks"] == [0, 1, 2]
          and preset["sd_noise_samples"] == 4 and preset["adapter_dropout"] == 0.2
          and preset["model_id"] == "facebook/dinov2-large")

    mix_model = build_model(make_cfg(fusion="adapter_mix"))
    mix_keys = trainable_state(mix_model)
    check("model adapter_mix: fusion.mix.* in the trainable state",
          "fusion.mix.weight" in mix_keys and "fusion.mix.bias" in mix_keys
          and "fusion.mix.weight" in mix_model.state_dict(), f"({len(mix_keys)} keys)")
    with torch.no_grad():
        diff = (mix_model(x) - none_feats).abs().max().item()
    check("model adapter_mix: features equal the fusion 'none' model's at init", diff < 1e-5, f"(max abs diff {diff:.2e})")
    check("preset unified_lora_large_sd_mix", get_config("unified_lora_large_sd_mix")["fusion"] == "adapter_mix")

    # Last: rounding through fp16 changes the weights of this model.
    fresh.sd.half()  # on CPU the _apply guard must bring the weights back to fp32
    check("sd: .half() on CPU keeps fp32 weights", all(p.dtype == torch.float32 for p in fresh.sd.parameters())
          and fresh.sd.prompt.dtype == torch.float32)

    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
