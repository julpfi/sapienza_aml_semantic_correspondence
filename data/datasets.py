"""Competition queries as a PyTorch dataset.

One row of train.csv / test.csv is one query. Every sample is a dict with these keys:

    query_id   str           competition `id` (one row of the submission)
    pair_id    str           image pair id (queries of the same pair share images)
    category   str           object category, or "all" when the CSV has none
    src_image  str           source image path relative to data_dir (also the cache key)
    trg_image  str           target image path relative to data_dir
    src_img    PIL.Image     source image (RGB)
    trg_img    PIL.Image     target image (RGB)
    src_xy     float32 [2]   query point, original source pixels
    trg_xys    float32 [N,2] accepted target points, original target pixels (None on test)
    scale      float         object size L used by PCK (None on test)

See data/README.md for the file layout.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from data.splits import load_val_pair_ids
from src.config import data_path

SPLITS = ("train", "val", "test")


def _open_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


class QueryDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.col = cfg["csv_columns"]
        self.image_root = Path(cfg["data_dir"])  # CSV image paths are relative to data_dir
        self.has_targets = self.col["trg_xys"] in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row, col = self.df.iloc[i], self.col
        src_image, trg_image = str(row[col["src_image"]]), str(row[col["trg_image"]])
        sample = {
            "query_id": str(row[col["query_id"]]),
            "pair_id": str(row[col["pair_id"]]) if col["pair_id"] in row.index else f"{src_image}|{trg_image}",
            "category": str(row[col["category"]]) if col["category"] in row.index else "all",
            "src_image": src_image,
            "trg_image": trg_image,
            "src_img": _open_rgb(self.image_root / src_image),
            "trg_img": _open_rgb(self.image_root / trg_image),
            "src_xy": np.asarray([row[col["src_x"]], row[col["src_y"]]], dtype=np.float32),
            "trg_xys": None,
            "scale": None,
        }
        if self.has_targets:
            trg_xys = np.asarray(json.loads(row[col["trg_xys"]]), dtype=np.float32).reshape(-1, 2)
            scale = float(row[col["scale"]])
            if len(trg_xys) == 0 or not np.isfinite(trg_xys).all():
                raise ValueError(f"Invalid {col['trg_xys']} for {sample['query_id']}")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid {col['scale']} for {sample['query_id']}")
            sample["trg_xys"], sample["scale"] = trg_xys, scale
        return sample


def read_queries(cfg, split):
    """Rows of one split as a DataFrame. train and val come from train.csv, split by image pair."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Use one of {SPLITS}.")
    if split == "test":
        df = pd.read_csv(data_path(cfg, "test_csv"))
        return df.head(cfg["max_test_queries"]) if cfg["max_test_queries"] else df

    train = pd.read_csv(data_path(cfg, "train_csv"))
    pair_col = cfg["csv_columns"]["pair_id"]
    val_ids = {str(p) for p in load_val_pair_ids(train, cfg)}
    is_val = train[pair_col].astype(str).isin(val_ids)
    if split == "val":
        # Same shuffle and cap as the Lang notebook.
        df = train.loc[is_val].sample(frac=1, random_state=cfg["split_seed"])
        return df.iloc[: cfg["max_val_queries"]] if cfg["max_val_queries"] else df
    df = train.loc[~is_val]
    return df.head(cfg["max_train_queries"]) if cfg["max_train_queries"] else df


def build_dataset(cfg, split):
    return QueryDataset(read_queries(cfg, split), cfg)


def collate_samples(batch):
    """Keep variable-size images and points as a plain list of sample dicts."""
    return batch
