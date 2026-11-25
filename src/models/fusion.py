"""Fusion of DINOv2 and Stable Diffusion features (both L2-normalised per cell).

"concat"      normalize(cat(dino, sd_concat_weight * sd)): no parameters, the frozen gate.
"adapter"     normalize(dino + adapter(sd)): 1x1 conv sd_dim -> dino_dim, adapter_layers residual
              bottleneck blocks, then a 1x1 conv whose weight and bias start at zero, so at
              initialisation fused == dino exactly and training starts from the DINOv2 model.
"adapter_mix" the same adapter s = adapter(sd), merged by a learned 1x1 conv:
              normalize(mix(cat(dino, s))), mix: 2 * dino_dim -> dino_dim initialised as [I | I]
              so update 0 == "adapter" == dino.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSIONS = ("concat", "adapter", "adapter_mix")


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
            self.sd_weight = cfg["sd_concat_weight"]
            return
        hidden = cfg["adapter_dim"]
        self.drop = nn.Dropout2d(cfg.get("adapter_dropout", 0.0))
        if cfg.get("adapter_bottleneck"):
            self.inp = nn.Conv2d(sd_dim, hidden, kernel_size=1)
            self.blocks = nn.Sequential(*[_Bottleneck(hidden, hidden) for _ in range(cfg["adapter_layers"])])
            self.out = nn.Conv2d(hidden, dino_dim, kernel_size=1)
        else:
            self.inp = nn.Conv2d(sd_dim, dino_dim, kernel_size=1)
            self.blocks = nn.Sequential(*[_Bottleneck(dino_dim, hidden) for _ in range(cfg["adapter_layers"])])
            self.out = nn.Conv2d(dino_dim, dino_dim, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if kind == "adapter_mix":
            self.mix = nn.Conv2d(2 * dino_dim, dino_dim, kernel_size=1)
            with torch.no_grad():
                eye = torch.eye(dino_dim).unsqueeze(-1).unsqueeze(-1)
                self.mix.weight.copy_(torch.cat([eye, eye], dim=1))
                self.mix.bias.zero_()

    def forward(self, dino_feats, sd_feats):
        """[B, C_dino, h, w] and [B, C_sd, h, w] -> fused [B, out_dim, h, w], L2-normalised."""
        if self.kind == "concat":
            return F.normalize(torch.cat([dino_feats, self.sd_weight * sd_feats], dim=1), dim=1)
        adapted = self.out(self.blocks(self.inp(self.drop(sd_feats))))
        if self.kind == "adapter_mix":
            return F.normalize(self.mix(torch.cat([dino_feats, adapted], dim=1)), dim=1)
        return F.normalize(dino_feats + adapted, dim=1)

