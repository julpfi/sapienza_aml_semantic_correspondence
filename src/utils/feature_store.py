"""On-disk cache of frozen backbone features.

With a frozen backbone, the features of an image never change, so the head can be trained
without running DINOv2 again. Features are stored per image and per setting, in a folder whose
name depends on everything that changes them (model, revision, layer, size, interpolation).
Only used when cache_features_to_disk is True and train_mode is "frozen".
"""

import hashlib
import json
from pathlib import Path


def feature_signature(cfg):
    keys = ["model_id", "model_revision", "feature_layer", "image_size", "image_mean", "image_std",
            "pos_embed_interp", "pos_embed_interp_offset"]
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


class FeatureStore:
    def __init__(self, cfg):
        self.root = Path(cfg["features_dir"]) / feature_signature(cfg)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, image_key, flip):
        name = hashlib.sha1(f"{image_key}|{int(flip)}".encode()).hexdigest()
        return self.root / name[:2] / f"{name}.pt"

    def get(self, image_key, flip=False):
        import torch

        path = self._path(image_key, flip)
        return torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None

    def put(self, image_key, feats, flip=False):
        import torch

        path = self._path(image_key, flip)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats.detach().cpu().half(), path)
