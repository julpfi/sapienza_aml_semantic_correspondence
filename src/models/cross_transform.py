"""Cross-image spatial context and multi-scale geometric adapter on top of backbone features.

Enabled with cfg["cross_transform"] = True (off when False).
Takes (src_feats, trg_feats), both [B, C, h, w] and L2-normalised, and enriches each side with:
1. Multi-scale 2D spatial context (dilations 1, 2, 3) so each cell knows its local part neighborhood
   across multiple camera distances (zoom levels).
2. Bidirectional cross-image attention with 2D rotary/sinusoidal positional encodings so the source
   and target feature maps exchange global scale, translation and rotation/pose context.

Both output projections are zero-initialised, so at update 0 the module is an exact identity and
training starts from the pretrained DINOv2 (or fused) baseline.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_2d(h, w, dim, device, dtype):
    """2D sinusoidal positional embedding [1, h*w, dim], normalised across the grid."""
    half = dim // 4
    if half == 0:
        return torch.zeros(1, h * w, dim, device=device, dtype=dtype)
    omega = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ang_y = grid_y.reshape(-1, 1) * omega[None] * math.pi
    ang_x = grid_x.reshape(-1, 1) * omega[None] * math.pi
    emb = torch.cat([ang_y.sin(), ang_y.cos(), ang_x.sin(), ang_x.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[None].to(dtype)


class CrossImageTransform(nn.Module):
    """Enrich (src_feats, trg_feats) [B, C, h, w] jointly with multi-scale + cross-attention context."""

    def __init__(self, in_dim, cfg):
        super().__init__()
        hidden = int(cfg["cross_dim"])
        heads = int(cfg["cross_heads"])
        dropout = float(cfg["cross_dropout"])
        if hidden % heads != 0:
            raise ValueError(f"cross_dim ({hidden}) must be divisible by cross_heads ({heads})")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.dropout = dropout

        # 1. Multi-scale spatial pyramid branch (camera-distance / zoom awareness: 3x3 at dilations 1, 2, 3)
        self.ms_in = nn.Conv2d(in_dim, hidden, kernel_size=1)
        self.ms_d1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, dilation=1, groups=heads)
        self.ms_d2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=heads)
        self.ms_d3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=3, dilation=3, groups=heads)
        self.ms_mix = nn.Sequential(nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(hidden * 3, hidden, kernel_size=1),
                                    nn.GELU())
        self.ms_out = nn.Conv2d(hidden, in_dim, kernel_size=1)
        nn.init.zeros_(self.ms_out.weight)
        nn.init.zeros_(self.ms_out.bias)

        # 2. Cross-image attention branch (rotation / viewpoint / relative layout awareness)
        self.norm_in = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.ca_drop = nn.Dropout(dropout)
        self.ca_out = nn.Linear(hidden, in_dim)
        nn.init.zeros_(self.ca_out.weight)
        nn.init.zeros_(self.ca_out.bias)

    def _multiscale(self, x):
        h = F.gelu(self.ms_in(x))
        cat = torch.cat([self.ms_d1(h), self.ms_d2(h), self.ms_d3(h)], dim=1)
        return self.ms_out(self.ms_mix(cat))

    def _attend(self, q_tok, kv_tok, pos):
        b, n, _ = q_tok.shape
        q = self.q_proj(q_tok + pos).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_tok + pos).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_tok).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        out = out.transpose(1, 2).reshape(b, n, self.hidden)
        return self.ca_out(self.ca_drop(F.gelu(out)))

    def forward(self, src_feats, trg_feats):
        """(src_feats, trg_feats), each [B, C, h, w] -> transformed L2-normalised (src_out, trg_out)."""
        b, c, h, w = src_feats.shape
        # Multi-scale spatial context on each image
        ms_src = self._multiscale(src_feats)
        ms_trg = self._multiscale(trg_feats)

        # Flatten to tokens [B, h*w, C] for bidirectional cross-attention
        s_tok = self.down(self.norm_in(src_feats.flatten(2).transpose(1, 2)))
        t_tok = self.down(self.norm_in(trg_feats.flatten(2).transpose(1, 2)))
        pos = _sinusoidal_2d(h, w, self.hidden, src_feats.device, s_tok.dtype)

        ca_src = self._attend(s_tok, t_tok, pos).transpose(1, 2).reshape(b, c, h, w)
        ca_trg = self._attend(t_tok, s_tok, pos).transpose(1, 2).reshape(b, c, h, w)

        src_out = F.normalize(src_feats + ms_src + ca_src, dim=1)
        trg_out = F.normalize(trg_feats + ms_trg + ca_trg, dim=1)
        return src_out, trg_out
