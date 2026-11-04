"""DINOv2 backbone loaded from the official Hugging Face weights (facebook/dinov2-*).

Equivalence with the Lang pipeline (torch.hub dinov2_vitb14), see docs/PRD_reorganization.md:
- Patch tokens of block `feature_layer` AFTER the final LayerNorm (= hub norm=True), CLS and
  register tokens dropped, L2-normalised per patch.
- With pos_embed_interp="hub", position embeddings are resized exactly like torch.hub
  (scale factor with a 0.1 offset). transformers resizes to the exact size instead, which
  changes features slightly at sizes other than 518 (cosine ~0.9997 at 644).
"""

import math
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_model_dir(cfg):
    """Folder with a local snapshot of model_id, e.g. weights/dinov2-base (see download_weights.py)."""
    return Path(cfg["weights_dir"]) / cfg["model_id"].split("/")[-1]


def _hub_interpolate_pos_encoding(self, embeddings, height, width):
    """torch.hub DINOv2 position-embedding interpolation, for a HF Dinov2Embeddings module."""
    num_patches = embeddings.shape[1] - 1
    num_positions = self.position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return self.position_embeddings
    pos_embed = self.position_embeddings.float()
    class_pos_embed = pos_embed[:, :1]
    patch_pos_embed = pos_embed[:, 1:]
    dim = embeddings.shape[-1]
    h0, w0 = height // self.patch_size, width // self.patch_size
    m = int(math.sqrt(num_positions))
    offset = self.hub_interp_offset
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, m, m, dim).permute(0, 3, 1, 2),
        scale_factor=((h0 + offset) / m, (w0 + offset) / m),
        mode="bicubic",
        antialias=False,
    )
    assert patch_pos_embed.shape[-2:] == (h0, w0)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(self.position_embeddings.dtype)


class DinoBackbone(nn.Module):
    """forward(x [B, 3, S, S] normalised) -> L2-normalised patch features [B, C, S/p, S/p]."""

    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel

        self.feature_layer = cfg["feature_layer"]
        local_dir = local_model_dir(cfg)
        if local_dir.is_dir():
            self.model = AutoModel.from_pretrained(local_dir)
        else:
            self.model = AutoModel.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
        config = self.model.config
        self.patch_size = config.patch_size
        self.num_registers = int(getattr(config, "num_register_tokens", 0) or 0)
        self.num_prefix = 1 + self.num_registers
        self.blocks = self.model.encoder.layer
        self.layernorm = self.model.layernorm
        if cfg["image_size"] % self.patch_size:
            raise ValueError(f"image_size {cfg['image_size']} must be a multiple of the patch size {self.patch_size}")

        # torch.hub models with registers already use the transformers behaviour (size, antialias).
        if cfg["pos_embed_interp"] == "hub" and self.num_registers == 0:
            emb = self.model.embeddings
            emb.hub_interp_offset = cfg["pos_embed_interp_offset"]
            emb.interpolate_pos_encoding = types.MethodType(_hub_interpolate_pos_encoding, emb)
        elif cfg["pos_embed_interp"] not in ("hub", "hf"):
            raise ValueError(f"pos_embed_interp must be 'hub' or 'hf', got {cfg['pos_embed_interp']!r}")

        self.freeze()

    # ---- what is trained ----
    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_last(self, n_blocks):
        """Make the last n blocks and the final LayerNorm trainable (finetune mode)."""
        for block in list(self.blocks)[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for p in self.layernorm.parameters():
            p.requires_grad_(True)

    def is_frozen(self):
        return not any(p.requires_grad for p in self.model.parameters())

    def train(self, mode=True):
        """Frozen parts always stay in eval mode; only trainable blocks follow `mode`."""
        super().train(mode)
        self.model.eval()
        if mode:
            for block in self.blocks:
                if any(p.requires_grad for p in block.parameters()):
                    block.train()
        return self

    def forward(self, x):
        size = x.shape[-1] // self.patch_size
        if self.feature_layer == -1:
            tokens = self.model(pixel_values=x).last_hidden_state  # already layer-normed
        else:
            hidden = self.model(pixel_values=x, output_hidden_states=True).hidden_states
            tokens = self.layernorm(hidden[self.feature_layer])
        tokens = tokens[:, self.num_prefix:, :]
        feats = tokens.transpose(1, 2).reshape(x.shape[0], -1, size, size)
        return F.normalize(feats, dim=1)
