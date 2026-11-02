"""Dataset adapters.

Everything downstream consumes the same sample dict:
    {
      "pair_id": str, "category": str,
      "src_img": PIL.Image, "trg_img": PIL.Image,
      "src_kps": float32 [K, 2]  (x, y in original source pixels),
      "trg_kps": float32 [K, 2] or None (unknown on the test split),
      "trg_bbox": float32 [4] (x1, y1, x2, y2) or None,
    }
If the competition ships a different format, write one new adapter that yields this dict.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class SPairDataset(Dataset):
    """SPair-71k layout: PairAnnotation/<split>/*.json and JPEGImages/<category>/<name>.jpg"""

    def __init__(self, root: str, split: str = "val", categories=None, max_pairs: int = 0):
        self.root = Path(root)
        ann_dir = self.root / "PairAnnotation" / split
        files = sorted(ann_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No annotation files in {ann_dir}")
        anns = [json.loads(f.read_text()) for f in files]
        if categories:
            anns = [a for a in anns if a["category"] in set(categories)]
        if max_pairs:
            # deterministic, category-balanced subsample so quick runs are representative
            rng = np.random.default_rng(0)
            anns = [anns[i] for i in sorted(rng.permutation(len(anns))[:max_pairs])]
        self.anns, self.files = anns, files

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, i):
        a = self.anns[i]
        cat = a["category"]
        img_dir = self.root / "JPEGImages" / cat
        return {
            "pair_id": a.get("filename", str(i)),
            "category": cat,
            "src_path": str(img_dir / a["src_imname"]), "trg_path": str(img_dir / a["trg_imname"]),
            "src_img": _rgb(img_dir / a["src_imname"]),
            "trg_img": _rgb(img_dir / a["trg_imname"]),
            "src_kps": np.asarray(a["src_kps"], dtype=np.float32).reshape(-1, 2),
            "trg_kps": np.asarray(a["trg_kps"], dtype=np.float32).reshape(-1, 2),
            "trg_bbox": np.asarray(a["trg_bndbox"], dtype=np.float32),
        }


def _parse_kps(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, str):
        v = json.loads(v) if v.strip().startswith("[") else ast.literal_eval(v)
    return np.asarray(v, dtype=np.float32).reshape(-1, 2)


class CSVPairDataset(Dataset):
    """Generic table adapter. Expected columns (rename in `colmap` to match the competition):
    pair_id, src_image, trg_image, category, src_kps, [trg_kps], [trg_bbox]
    Keypoints/bbox columns are JSON lists, e.g. "[[12.0, 40.5], [80.1, 33.0]]".
    One-keypoint-per-row query files (like the SOCO test set: id -> one prediction) are supported:
    use pair_id = query id and either src_kps="[x, y]" or separate src_x / src_y columns.
    """

    def __init__(self, csv_path: str, image_dir: str, max_pairs: int = 0, colmap: dict | None = None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        if max_pairs:
            self.df = self.df.head(max_pairs)
        self.image_dir = Path(image_dir)
        self.c = {k: k for k in ["pair_id", "src_image", "trg_image", "category", "src_kps", "trg_kps", "trg_bbox"]}
        self.c.update(colmap or {})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r, c = self.df.iloc[i], self.c
        get = lambda k: r[c[k]] if c[k] in r.index else None
        bbox = get("trg_bbox")
        src_kps = _parse_kps(get("src_kps")) if c["src_kps"] in r.index else \
            np.asarray([[r["src_x"], r["src_y"]]], dtype=np.float32)
        trg_kps = _parse_kps(get("trg_kps")) if c["trg_kps"] in r.index else \
            (np.asarray([[r["trg_x"], r["trg_y"]]], dtype=np.float32) if "trg_x" in r.index else None)
        return {
            "pair_id": str(get("pair_id")),
            "category": str(get("category") or "all"),
            "src_path": str(self.image_dir / get("src_image")), "trg_path": str(self.image_dir / get("trg_image")),
            "src_img": _rgb(self.image_dir / get("src_image")),
            "trg_img": _rgb(self.image_dir / get("trg_image")),
            "src_kps": src_kps,
            "trg_kps": trg_kps,
            "trg_bbox": None if bbox is None else np.asarray(_parse_kps(bbox)).reshape(-1)[:4],
        }


def build_dataset(cfg, split: str | None = None):
    split = split or cfg.split
    if cfg.data_format == "spair":
        return SPairDataset(cfg.data_root, split, cfg.categories, cfg.max_pairs)
    if cfg.data_format == "csv":
        return CSVPairDataset(cfg.csv_path, cfg.image_dir, cfg.max_pairs)
    raise ValueError(cfg.data_format)


def collate_keep(batch):
    """Keep variable-size images and keypoints as lists."""
    return batch
