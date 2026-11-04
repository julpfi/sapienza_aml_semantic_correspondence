"""Trainable head on top of the frozen backbone features.

"none"   identity: reproduces the training-free Lang pipeline exactly.
"linear" 1x1 convolution, initialised as the identity (when head_dim equals the feature size),
         so training starts from the Lang behaviour.
"mlp"    residual 2-layer MLP; the last layer starts at zero, so it also starts as the identity.
The output is always L2-normalised per patch, because matching uses cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ("none", "linear", "mlp")


class ProjectionHead(nn.Module):
    def __init__(self, kind, in_dim, out_dim):
        super().__init__()
        if kind not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {kind!r}")
        self.kind = kind
        if kind == "linear":
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
            with torch.no_grad():
                nn.init.zeros_(self.proj.bias)
                if in_dim == out_dim:
                    self.proj.weight.copy_(torch.eye(in_dim).view(in_dim, in_dim, 1, 1))
        elif kind == "mlp":
            self.fc1 = nn.Conv2d(in_dim, in_dim, kernel_size=1)
            self.fc2 = nn.Conv2d(in_dim, in_dim, kernel_size=1)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, feats):
        if self.kind == "none":
            return feats
        if self.kind == "linear":
            out = self.proj(feats)
        else:
            out = feats + self.fc2(F.gelu(self.fc1(feats)))
        return F.normalize(out, dim=1)
