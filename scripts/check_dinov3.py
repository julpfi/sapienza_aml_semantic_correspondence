"""Check the DINOv3 backbone (src/models/backbone_dinov3.py) on CPU, without the gated weights.

    python scripts/check_dinov3.py

A tiny DINOv3 and a tiny DINOv2 with random weights are built from the transformers classes and saved as
local snapshots (<weights_dir>/<model folder>), so the real loading path of both backbones is exercised.
Checks: backbone choice by model_id, features equal to the transformers forward, prefix (CLS + registers)
dropped, feature_layer indexing, RoPE kept deterministic in training mode, any grid without interpolation,
patch-size check, finetune / LoRA trainable sets and their optimiser groups, gradients, checkpoint round trip,
end-to-end predict, and the DINOv2 path unchanged.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import DINOV3_B16, get_config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.backbone import DinoBackbone  # noqa: E402
from src.models.backbone_dinov3 import Dinov3Backbone, build_backbone, is_dinov3  # noqa: E402
from src.models.lora import LoRALinear, lora_parameters  # noqa: E402
from src.train import configure_trainable  # noqa: E402
from src.utils.checkpoint import load_model_weights, trainable_state  # noqa: E402

failures = []
DIM, LAYERS, HEADS, REGISTERS = 64, 4, 4, 4


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


def save_tiny_models(weights_dir):
    """Random tiny DINOv3 / DINOv2 snapshots under the folder names local_model_dir expects."""
    from transformers import Dinov2Config, Dinov2Model, DINOv3ViTConfig, DINOv3ViTModel

    torch.manual_seed(0)
    v3 = DINOv3ViTModel(DINOv3ViTConfig(
        hidden_size=DIM, intermediate_size=2 * DIM, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
        patch_size=16, num_register_tokens=REGISTERS, key_bias=False, layerscale_value=1.0,
        pos_embed_rescale=2.0))  # the released models keep this pretraining augmentation on in training mode
    v3.save_pretrained(Path(weights_dir) / DINOV3_B16["model_id"].split("/")[-1])
    v2 = Dinov2Model(Dinov2Config(hidden_size=DIM, intermediate_size=2 * DIM, num_hidden_layers=LAYERS,
                                  num_attention_heads=HEADS, patch_size=14, image_size=518))
    v2.save_pretrained(Path(weights_dir) / "dinov2-base")


def make_cfg(weights_dir, dinov3=True, **extra):
    cfg = get_config()
    cfg.update({"weights_dir": str(weights_dir), "device": "cpu", "fusion": "none", "train_mode": "finetune",
                "unfreeze_blocks": 2, "lora_blocks": 2, "feature_layer": -1})
    if dinov3:
        cfg.update(DINOV3_B16)
        cfg["image_size"] = 96  # 6 x 6 cells: small and fast
    else:
        cfg["image_size"] = 84
    cfg.update(extra)
    return cfg


def images(size, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, size, size, generator=g)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        save_tiny_models(tmp)
        run_checks(tmp)
    print(f"\n{len(failures)} failure(s)" if failures else "\nALL PASS")
    return 1 if failures else 0


def run_checks(weights_dir):
    # --- backbone choice ---
    check("is_dinov3 recognises DINOv3 ids only",
          is_dinov3(DINOV3_B16["model_id"]) and not is_dinov3("facebook/dinov2-base")
          and not is_dinov3("facebook/dinov2-large"))
    cfg = make_cfg(weights_dir)
    bb = build_backbone(cfg).eval()
    check("build_backbone: DINOv3 model_id -> Dinov3Backbone", type(bb) is Dinov3Backbone)
    bb2 = build_backbone(make_cfg(weights_dir, dinov3=False)).eval()
    check("build_backbone: DINOv2 model_id -> DinoBackbone (unchanged class)", type(bb2) is DinoBackbone)
    check("DINOv3: patch 16, CLS + 4 registers as prefix", bb.patch_size == 16 and bb.num_prefix == 1 + REGISTERS,
          f"(patch {bb.patch_size}, prefix {bb.num_prefix})")
    check("DINOv3: blocks / layernorm are the transformers modules",
          bb.blocks is bb.model.model.layer and bb.layernorm is bb.model.norm and len(bb.blocks) == LAYERS)
    check("everything frozen after construction", not any(p.requires_grad for p in bb.parameters()))

    # --- features ---
    x = images(96)
    with torch.no_grad():
        feats = bb(x)
        ref = bb.model(pixel_values=x).last_hidden_state[:, bb.num_prefix:]
    ref = F.normalize(ref.transpose(1, 2).reshape(2, DIM, 6, 6), dim=1)
    check("forward: [B, C, S/16, S/16]", tuple(feats.shape) == (2, DIM, 6, 6), f"{tuple(feats.shape)}")
    check("forward: unit norm per cell", torch.allclose(feats.norm(dim=1), torch.ones(2, 6, 6), atol=1e-5))
    check("forward = transformers last_hidden_state minus prefix, L2-normalised",
          torch.allclose(feats, ref, atol=1e-5), f"(max diff {(feats - ref).abs().max():.2e})")
    with torch.no_grad():
        manual = bb._tokens_after(x, LAYERS)
        official = bb.model(pixel_values=x).last_hidden_state
    check("manual block loop over all blocks = transformers forward", torch.allclose(manual, official, atol=1e-5),
          f"(max diff {(manual - official).abs().max():.2e})")
    with torch.no_grad():
        early = Dinov3Backbone(make_cfg(weights_dir, feature_layer=-2)).eval().tokens(x)
        direct = bb._tokens_after(x, LAYERS - 1)
    check("feature_layer -2 = tokens after the second-to-last block", torch.allclose(early, direct, atol=1e-6)
          and not torch.allclose(early, official, atol=1e-3))
    check("feature_layer outside the blocks is refused",
          raises(lambda: Dinov3Backbone(make_cfg(weights_dir, feature_layer=LAYERS + 1)).tokens(x), ValueError))
    with torch.no_grad():
        cls, cls_map = bb.cls(x)
    check("cls: [B, C] token and [B, h, w] map", tuple(cls.shape) == (2, DIM) and tuple(cls_map.shape) == (2, 6, 6))

    # --- position encoding (RoPE) ---
    with torch.no_grad():
        other_grids = [tuple(bb(images(s, n=1)).shape[-2:]) for s in (64, 160, 224)]
    check("RoPE: any multiple of 16 works without interpolation", other_grids == [(4, 4), (10, 10), (14, 14)],
          f"{other_grids}")
    check("image_size not a multiple of 16 is refused",
          raises(lambda: Dinov3Backbone(make_cfg(weights_dir, image_size=644)), ValueError))
    bb.unfreeze_last(2)
    bb.train()
    check("train(): RoPE module and frozen blocks stay in eval, trained blocks train",
          not bb.model.rope_embeddings.training and not bb.blocks[0].training and bb.blocks[-1].training
          and bb.blocks[-2].training)
    rope = bb.model.rope_embeddings
    with torch.no_grad():
        a, b = bb(x), bb(x)
        cos_used = rope(x)[0]
        rope.eval()
        cos_eval = rope(x)[0]
        torch.manual_seed(0)  # the augmentation draws a random scale: keep this check reproducible
        rope.train()  # what would happen without the eval rule (the random tiny model barely uses positions,
        cos_aug = rope(x)[0]  # so this is checked on the RoPE angles, not on the features)
        rope.eval()
    check("train mode: features deterministic", torch.equal(a, b))
    check("train mode: RoPE positions = the eval positions (no coordinate augmentation)", torch.equal(cos_used, cos_eval))
    check("(sensitivity) RoPE in training mode would move the positions", not torch.allclose(cos_aug, cos_eval, atol=1e-3),
          f"(max diff {(cos_aug - cos_eval).abs().max():.2f})")
    bb.freeze()
    bb.eval()

    # --- what is trained: finetune ---
    model = build_model(make_cfg(weights_dir))
    groups = configure_trainable(model, model.cfg)
    names = [g["name"] for g in groups]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = {f"backbone.model.model.layer.{i}." for i in (LAYERS - 2, LAYERS - 1)} | {"backbone.model.norm."}
    check("finetune: groups = last two blocks + final LayerNorm", names == [f"block_{LAYERS - 1}", f"block_{LAYERS - 2}",
                                                                           "layernorm"], f"{names}")
    check("finetune: only those parameters are trainable",
          trainable and all(any(n.startswith(e) for e in expected) for n in trainable)
          and all(any(n.startswith(e) for n in trainable) for e in expected))
    lr = {g["name"]: g["lr"] for g in groups}
    check("finetune: layer decay applied", abs(lr[f"block_{LAYERS - 2}"] - model.cfg["lr_backbone"] * model.cfg["layer_decay"])
          < 1e-12)
    model.train()
    sample = model.backbone(images(96, n=1))
    sample.sum().backward()
    top, bottom = model.backbone.blocks[-1].attention.q_proj.weight.grad, model.backbone.blocks[0].attention.q_proj.weight.grad
    check("finetune: gradients reach the top block, not the frozen ones", top is not None and top.abs().sum() > 0
          and bottom is None)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    state = trainable_state(model)
    fresh = load_model_weights(build_model(make_cfg(weights_dir)), state)
    with torch.no_grad():
        same = torch.allclose(model.eval()(x), fresh(x), atol=1e-6)
    check("finetune: checkpoint round trip gives the same features", same)

    # --- what is trained: LoRA ---
    frozen = build_model(make_cfg(weights_dir))
    lora = build_model(make_cfg(weights_dir, train_mode="lora"))
    groups = configure_trainable(lora, lora.cfg)
    wrapped = [b.attention.q_proj for b in lora.backbone.blocks[-2:]] + [b.attention.v_proj for b in lora.backbone.blocks[-2:]]
    check("lora: q_proj / v_proj of the last lora_blocks blocks wrapped",
          all(isinstance(m, LoRALinear) for m in wrapped)
          and not isinstance(lora.backbone.blocks[0].attention.q_proj, LoRALinear)
          and not isinstance(lora.backbone.blocks[-1].attention.k_proj, LoRALinear))
    n_lora = sum(p.numel() for p in lora_parameters(lora))
    check("lora: one group, only the LoRA weights trainable",
          [g["name"] for g in groups] == ["lora"]
          and {n for n, p in lora.named_parameters() if p.requires_grad} == {n for n, p in lora.named_parameters()
                                                                            if "lora_" in n},
          f"({n_lora} parameters)")
    check("lora: parameter count = blocks x 2 layers x 2 x rank x dim",
          n_lora == 2 * 2 * 2 * lora.cfg["lora_rank"] * DIM, f"({n_lora})")
    with torch.no_grad():
        diff = (frozen(x) - lora.eval()(x)).abs().max()
    check("lora: at init the features equal the frozen DINOv3", diff < 1e-6, f"(max diff {diff:.2e})")

    # --- end to end ---
    rng = np.random.default_rng(0)
    sample = {"src_img": Image.fromarray(rng.integers(0, 255, (120, 90, 3), dtype=np.uint8)),
              "trg_img": Image.fromarray(rng.integers(0, 255, (80, 140, 3), dtype=np.uint8)),
              "src_image": "a.jpg", "trg_image": "b.jpg", "src_xy": np.array([40.0, 60.0], dtype=np.float32)}
    pred, flipped = frozen.predict(sample, cache=frozen.new_cache())
    check("predict: finite point inside the target image", np.isfinite(pred).all() and 0 <= pred[0] <= 140
          and 0 <= pred[1] <= 80, f"({pred})")
    emb, emb_map = frozen.cls_embedding(sample["src_img"])
    check("cls_embedding: [C] and [h, w] with the eval preprocessing", tuple(emb.shape) == (DIM,)
          and tuple(emb_map.shape) == (6, 6))

    # --- DINOv2 path unchanged ---
    v2 = build_model(make_cfg(weights_dir, dinov3=False))
    with torch.no_grad():
        f2 = v2(images(84, n=1))
    check("DINOv2: still DinoBackbone with patch 14 and hub position-embedding interpolation",
          type(v2.backbone) is DinoBackbone and v2.backbone.patch_size == 14
          and hasattr(v2.backbone.model.embeddings, "hub_interp_offset") and tuple(f2.shape) == (1, DIM, 6, 6))


if __name__ == "__main__":
    sys.exit(main())
