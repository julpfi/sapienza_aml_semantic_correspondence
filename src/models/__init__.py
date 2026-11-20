"""Correspondence model: DINOv2 backbone + head (+ optional Stable Diffusion fusion) + matching.

Every prediction is (x, y) in original target-image pixels.
"""

import torch
import torch.nn as nn

from data.transforms import preprocess_image
from src.config import resolved_device
from src.models.backbone import DinoBackbone
from src.models.head import ProjectionHead
from src.models.lora import apply_lora
from src.models.matching import FeatureCache, choose_source, predict_from_features


class CorrespondenceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = DinoBackbone(cfg)
        if cfg["train_mode"] == "lora":
            apply_lora(self.backbone, cfg)
        in_dim = self.backbone.model.config.hidden_size
        out_dim = in_dim if cfg["head"] == "none" or cfg.get("head_dim") is None else cfg["head_dim"]
        self.head = ProjectionHead(cfg["head"], in_dim, out_dim)
        # Imported only here, so fusion "none" never imports diffusers and keeps the v2 state-dict keys.
        fusion = cfg["fusion"]
        if fusion != "none":
            from src.models.fusion import Fusion
            from src.models.sd import SDFeatures

            self.sd = SDFeatures(cfg)
            self.fusion = Fusion(fusion, out_dim, self.sd.out_dim, cfg)
            out_dim = self.fusion.out_dim
        if cfg.get("cross_transform"):
            from src.models.cross_transform import CrossImageTransform

            self.cross_transform = CrossImageTransform(out_dim, cfg)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def forward(self, x):
        """Normalised images [B, 3, S, S] -> matching features [B, C, h, w]."""
        feats = self.head(self.backbone(x))
        if hasattr(self, "fusion"):
            feats = self.fusion(feats, self.sd(x, feats.shape[-2:]))
        return feats

    def pair_features(self, src_feats, trg_feats):
        """Apply cross_transform (when enabled) to a pair of single-image feature maps [C, h, w] or batches [B, C, h, w]."""
        if not hasattr(self, "cross_transform"):
            return src_feats, trg_feats
        if src_feats.ndim == 3:
            s, t = self.cross_transform(src_feats[None], trg_feats[None])
            return s[0], t[0]
        return self.cross_transform(src_feats, trg_feats)

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

    @torch.no_grad()
    def cls_embedding(self, img):
        """DINOv2 CLS token [C] (L2-normalised) of one PIL image and its CLS-patch cosine map [h, w],
        with the eval preprocessing."""
        cls, cls_map = self.backbone.cls(preprocess_image(img, self.cfg).to(self.device))
        return cls[0], cls_map[0]

    def new_cache(self):
        return FeatureCache(self.cfg["feature_cache_size"])

    @torch.no_grad()
    def predict(self, sample, cfg=None, cache=None):
        """Return (pred_xy [2] in original target pixels, used_flip) for one sample."""
        cfg = cfg or self.cfg
        src_feats = self.image_features(sample["src_img"], cache=cache, key=sample["src_image"])
        trg_feats = self.image_features(sample["trg_img"], cache=cache, key=sample["trg_image"])
        src_feats, trg_raw = self.pair_features(src_feats, trg_feats)
        src_flip, trg_flip = None, trg_raw
        if cfg["flip_test"]:
            src_flip_base = self.image_features(sample["src_img"], flip=True, cache=cache, key=sample["src_image"])
            src_flip, trg_flip = self.pair_features(src_flip_base, trg_feats)
        src_chosen, flipped = choose_source(src_feats, src_flip, trg_raw)
        trg_chosen = trg_flip if flipped else trg_raw
        if cfg.get("constellation_align"):
            from src.models.constellation import load_pair_source_xy, predict_with_constellation

            ctx_xy = load_pair_source_xy(cfg, sample["pair_id"], fallback_xy=sample["src_xy"][None])
            pred, _ = predict_with_constellation(src_chosen, trg_chosen, sample["src_xy"][None], ctx_xy,
                                                 sample["src_img"].size, sample["trg_img"].size, cfg, flipped)
            return pred[0], flipped
        pred = predict_from_features(src_chosen, trg_chosen, sample["src_xy"][None], sample["src_img"].size,
                                     sample["trg_img"].size, cfg, flipped)
        return pred[0], flipped


def build_model(cfg, device=None):
    """Build the model on `device` (default from cfg), frozen and in eval mode."""
    model = CorrespondenceModel(cfg).to(device or resolved_device(cfg))
    return model.eval()
