"""Frozen (or partially unfrozen) DINO feature extractor with one interface for torch.hub and HF."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoFeatures(nn.Module):
    """forward(x [B,3,S,S] normalized) -> L2-normalized patch features [B, C, S/p, S/p]."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.source = cfg.backbone_source
        if self.source == "hub":
            kw = dict(source="local") if cfg.hub_local else {}
            # pretrained=False when we ship weights ourselves (offline Kaggle submissions)
            self.model = torch.hub.load(cfg.hub_repo, cfg.backbone, pretrained=not cfg.weights_path, **kw)
            if cfg.weights_path:
                sd = torch.load(cfg.weights_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(sd)
            self.patch = self.model.patch_size if hasattr(self.model, "patch_size") else self.model.patch_embed.patch_size[0]
            self.blocks = self.model.blocks
        elif self.source == "hf":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(cfg.backbone)
            hc = self.model.config
            self.patch = hc.patch_size
            self.n_prefix = 1 + int(getattr(hc, "num_register_tokens", 0) or 0)
            enc = getattr(self.model, "encoder", None)
            self.blocks = enc.layer if enc is not None else self.model.layer
        else:
            raise ValueError(self.source)
        if cfg.img_size % self.patch:
            raise ValueError(f"img_size {cfg.img_size} must be a multiple of patch size {self.patch}")
        if cfg.ft_checkpoint:
            sd = torch.load(cfg.ft_checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(sd, strict=False)
        self.freeze_all()

    # ---- trainability helpers (stage 2) ----
    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_last(self, n: int):
        for blk in list(self.blocks)[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        norm = getattr(self.model, "norm", None)
        if norm is None:
            norm = getattr(self.model, "layernorm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = w = x.shape[-1] // self.patch
        if self.source == "hub":
            n_blocks = len(self.blocks)
            idx = self.cfg.layer % n_blocks
            feats = self.model.get_intermediate_layers(x, n=[idx], reshape=True, norm=True)[0]
        else:
            out = self.model(pixel_values=x, output_hidden_states=True)
            tok = out.hidden_states[self.cfg.layer]           # [B, prefix + h*w, C]
            tok = tok[:, self.n_prefix:, :]
            feats = tok.transpose(1, 2).reshape(x.shape[0], -1, h, w)
        return F.normalize(feats, dim=1)


def preprocess(img: Image.Image, size: int, flip: bool = False) -> torch.Tensor:
    """PIL -> normalized [1,3,size,size]. Square resize; keypoints are rescaled accordingly."""
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (t - IMAGENET_MEAN) / IMAGENET_STD
