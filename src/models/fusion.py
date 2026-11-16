"""Fusion of DINOv2 and Stable Diffusion features (both L2-normalised per cell).

"concat"  normalize(cat(dino, sd_concat_weight * sd)): no parameters, the frozen gate.
"adapter" normalize(dino + adapter(sd)): 1x1 conv sd_dim -> dino_dim, adapter_layers residual
          bottleneck blocks, then a 1x1 conv whose weight and bias start at zero, so at
          initialisation fused == dino exactly and training starts from the DINOv2 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSIONS = ("concat", "adapter")


class _Bottleneck(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1))

    def forward(self, x):
        return x + self.body(x)


class Fusion(nn.Module):
    def __init__(self, kind, dino_dim, sd_dim, cfg):
        super().__init__()
        if kind not in FUSIONS:
            raise ValueError(f"fusion must be 'none' or one of {FUSIONS}, got {kind!r}")
        self.kind = kind
        self.out_dim = dino_dim + sd_dim if kind == "concat" else dino_dim
        if kind == "concat":
            self.sd_weight = cfg.get("sd_concat_weight", 0.5)
            return
        hidden = cfg.get("adapter_dim", 256)
        self.inp = nn.Conv2d(sd_dim, dino_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[_Bottleneck(dino_dim, hidden) for _ in range(cfg.get("adapter_layers", 2))])
        self.out = nn.Conv2d(dino_dim, dino_dim, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, dino_feats, sd_feats):
        """[B, C_dino, h, w] and [B, C_sd, h, w] -> fused [B, out_dim, h, w], L2-normalised."""
        if self.kind == "concat":
            return F.normalize(torch.cat([dino_feats, self.sd_weight * sd_feats], dim=1), dim=1)
        return F.normalize(dino_feats + self.out(self.blocks(self.inp(sd_feats))), dim=1)

