"""DINOv3 ViT backbone (facebook/dinov3-*), with the same interface as DinoBackbone (DINOv2).

Which backbone a run uses follows from model_id: build_backbone picks this class when the id contains
"dinov3", so every DINOv2 run (and its saved config.json) keeps loading DinoBackbone exactly as before.

What differs from DINOv2 (transformers DINOv3ViTModel, checked on transformers 5.22.1):
- Module names: blocks are model.model.layer (DINOv2: model.encoder.layer), the final LayerNorm is
  model.norm (DINOv2: model.layernorm). Attention keeps q_proj / k_proj / v_proj / o_proj, so LoRA
  (src/models/lora.py) wraps the same layers without changes.
- Patch size 16 (DINOv2: 14): image_size must be a multiple of 16. 736 px -> 46 x 46 cells, the grid of
  DINOv2 at 644, so matching settings measured in cells (soft_window, marginal_radius, label_jitter,
  hard_neg_*) cover the same share of the image.
- Tokens: CLS + 4 register tokens + patches. The prefix is dropped like the CLS token of DINOv2.
- Position encoding: RoPE (rotary) instead of a learned position-embedding table. Each patch gets its
  centre coordinate normalised to [-1, 1] over the patch grid, and the queries/keys of the patch tokens are
  rotated by angles derived from it. There is no table to interpolate, so any grid works natively and
  pos_embed_interp / pos_embed_interp_offset are not used.
  In training mode the RoPE module would randomly rescale those coordinates (pretraining augmentation,
  config.pos_embed_rescale). DinoBackbone.train keeps every module that is not a trainable block in eval
  mode, so the RoPE coordinates are always the deterministic eval ones, in training and inference alike.
- Image normalisation: the LVD-1689M models use the ImageNet mean/std, the image_mean / image_std of BASE.
"""

import torch.nn as nn

from src.models.backbone import DinoBackbone, local_model_dir


def is_dinov3(model_id):
    return "dinov3" in model_id.lower()


def build_backbone(cfg):
    """DINOv3 backbone when model_id is a DINOv3 model, else the DINOv2 backbone (unchanged)."""
    return Dinov3Backbone(cfg) if is_dinov3(cfg["model_id"]) else DinoBackbone(cfg)


class Dinov3Backbone(DinoBackbone):
    """forward(x [B, 3, S, S] normalised) -> L2-normalised patch features [B, C, S/16, S/16].

    Inherited unchanged from DinoBackbone: freeze, unfreeze_last, train, cls, forward. They only use
    the attributes set here (model, blocks, layernorm, patch_size, num_prefix) and tokens().
    """

    def __init__(self, cfg):
        nn.Module.__init__(self)  # DinoBackbone.__init__ loads DINOv2 and patches its position embeddings
        from transformers import AutoModel


        self.feature_layer = cfg["feature_layer"]
        local_dir = local_model_dir(cfg)
        attn = {"attn_implementation": cfg["attn_implementation"]} if cfg.get("attn_implementation") else {}
        if local_dir.is_dir():
            self.model = AutoModel.from_pretrained(local_dir, **attn)
        else:  # gated on the Hub: needs HF_TOKEN of an account that accepted the DINOv3 license
            self.model = AutoModel.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], **attn)
        config = self.model.config
        if config.model_type != "dinov3_vit":
            raise ValueError(f"{cfg['model_id']} is not a DINOv3 ViT (model_type {config.model_type!r}).")
        self.patch_size = config.patch_size
        self.num_registers = int(getattr(config, "num_register_tokens", 0) or 0)
        self.num_prefix = 1 + self.num_registers
        self.blocks = self.model.model.layer
        self.layernorm = self.model.norm
        if cfg["image_size"] % self.patch_size:
            raise ValueError(f"image_size {cfg['image_size']} must be a multiple of the patch size {self.patch_size}")
        self.freeze()


    def tokens(self, x):
        """All layer-normed tokens [B, prefix + patches, C] of block `feature_layer` (CLS first)."""
        if self.feature_layer == -1:
            return self.model(pixel_values=x).last_hidden_state  # already layer-normed
        n_blocks = self.feature_layer if self.feature_layer >= 0 else len(self.blocks) + 1 + self.feature_layer
        return self._tokens_after(x, n_blocks)


    def _tokens_after(self, x, n_blocks):
        """Tokens after the first n_blocks blocks and the final LayerNorm (DINOv2 hidden_states[n] indexing:
        n = 0 is the embedding output, n = len(blocks) the last block)."""
        if not 0 <= n_blocks <= len(self.blocks):
            raise ValueError(f"feature_layer {self.feature_layer} is outside the {len(self.blocks)} blocks")
        x = x.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden = self.model.embeddings(x)
        position_embeddings = self.model.rope_embeddings(x)
        for block in list(self.blocks)[:n_blocks]:
            hidden = block(hidden, position_embeddings=position_embeddings)
        return self.layernorm(hidden)
