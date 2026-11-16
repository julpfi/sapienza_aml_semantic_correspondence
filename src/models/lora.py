"""LoRA (low-rank adaptation) of the attention query/value linears of the last DINOv2 blocks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen nn.Linear plus a trainable low-rank update: base(x) + alpha/rank * dropout(x) A^T B^T."""

    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)


def lora_parameters(model):
    """All LoRA parameters of `model` (lora_A / lora_B), as a list."""
    return [p for name, p in model.named_parameters() if "lora_A" in name or "lora_B" in name]


def apply_lora(backbone, cfg):
    """Wrap attention q_proj and v_proj of the last cfg["lora_blocks"] blocks; return the LoRA parameters.

    The wrapped layer equals the base layer at init. Idempotent. The new parameters are created with
    requires_grad True; DinoBackbone.freeze() turns them off, so the trainer re-enables lora_parameters.
    """
    for block in list(backbone.blocks)[-cfg["lora_blocks"]:]:
        attn = block.attention  # transformers 5.x: q_proj / k_proj / v_proj / o_proj directly on the attention
        for name in ("q_proj", "v_proj"):
            layer = getattr(attn, name)
            if not isinstance(layer, LoRALinear):
                setattr(attn, name, LoRALinear(layer, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"]))
    return lora_parameters(backbone)
