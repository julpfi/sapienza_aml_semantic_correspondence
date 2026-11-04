"""Correspondence model: DINOv2 backbone + head + matching.

Every prediction is (x, y) in original target-image pixels.
"""

import torch
import torch.nn as nn

from data.transforms import preprocess_image
from src.config import resolved_device
from src.models.backbone import DinoBackbone
from src.models.head import ProjectionHead
from src.models.matching import FeatureCache, alignment_score, predict_from_features


class CorrespondenceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = DinoBackbone(cfg)
        in_dim = self.backbone.model.config.hidden_size
        out_dim = in_dim if cfg["head"] == "none" else cfg["head_dim"]
        self.head = ProjectionHead(cfg["head"], in_dim, out_dim)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def forward(self, x):
        """Normalised images [B, 3, S, S] -> matching features [B, C, h, w]."""
        return self.head(self.backbone(x))

    @torch.no_grad()
    def image_features(self, img, flip=False, cache=None, key=None):
        """Features [C, h, w] of one PIL image, reusing `cache` when the image was seen."""
        if cache is not None and key is not None:
            hit = cache.get((key, flip))
            if hit is not None:
                return hit
        feats = self(preprocess_image(img, self.cfg, flip).to(self.device))[0]
        if cache is not None and key is not None:
            cache.put((key, flip), feats)
        return feats

    def new_cache(self):
        return FeatureCache(self.cfg["feature_cache_size"])

    @torch.no_grad()
    def predict(self, sample, cfg=None, cache=None):
        """Return (pred_xy [2] in original target pixels, used_flip) for one sample."""
        cfg = cfg or self.cfg
        src_feats = self.image_features(sample["src_img"], cache=cache, key=sample["src_image"])
        trg_feats = self.image_features(sample["trg_img"], cache=cache, key=sample["trg_image"])
        flipped = False
        if cfg["flip_test"]:
            src_flip = self.image_features(sample["src_img"], flip=True, cache=cache, key=sample["src_image"])
            if alignment_score(src_flip, trg_feats) > alignment_score(src_feats, trg_feats):
                src_feats, flipped = src_flip, True
        pred = predict_from_features(src_feats, trg_feats, sample["src_xy"][None], sample["src_img"].size,
                                     sample["trg_img"].size, cfg, flipped)
        return pred[0], flipped


def build_model(cfg, device=None):
    """Build the model on `device` (default from cfg), frozen and in eval mode."""
    model = CorrespondenceModel(cfg).to(device or resolved_device(cfg))
    return model.eval()
