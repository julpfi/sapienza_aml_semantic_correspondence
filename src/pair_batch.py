"""Pair-batched training data: one item = one image pair with all its queries (cfg["pair_batch"]).

Built from the same DataFrame as QueryDataset (read_queries in data/datasets.py), so the two see
the same queries. An item is a dict:

    pair_id    str
    category   str
    src_image  str               source image path relative to data_dir (also the cache key)
    trg_image  str
    src_img    PIL.Image         source image (RGB), opened once per pair
    trg_img    PIL.Image
    query_ids  list of K str
    src_xys    float32 [K, 2]    query points, original source pixels
    trg_xys    list of K float32 [N_k, 2]   accepted targets, original target pixels
    scale      list of K float   object size L of each query

Pairs come in the order of their first row in the DataFrame; queries keep the DataFrame order.
"""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from data.datasets import _open_rgb


class PairDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.col = cfg["csv_columns"]
        self.image_root = Path(cfg["data_dir"])
        if self.col["trg_xys"] not in self.df.columns:
            raise ValueError("PairDataset needs accepted targets (training rows).")
        col = self.col
        if col["pair_id"] in self.df.columns:
            keys = self.df[col["pair_id"]].astype(str)
        else:
            keys = self.df[col["src_image"]].astype(str) + "|" + self.df[col["trg_image"]].astype(str)
        rows = {}
        for i, key in enumerate(keys):
            rows.setdefault(key, []).append(i)  # dicts keep insertion order = first appearance
        self.pair_ids, self.rows = list(rows), list(rows.values())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        col, rows = self.col, self.df.iloc[self.rows[i]]
        first = rows.iloc[0]
        src_image, trg_image = str(first[col["src_image"]]), str(first[col["trg_image"]])
        same = (rows[col["src_image"]].astype(str) == src_image) & (rows[col["trg_image"]].astype(str) == trg_image)
        if not same.all():
            raise ValueError(f"Pair {self.pair_ids[i]} has rows with different images.")
        item = {
            "pair_id": self.pair_ids[i],
            "category": str(first[col["category"]]) if col["category"] in rows.columns else "all",
            "src_image": src_image,
            "trg_image": trg_image,
            "src_img": _open_rgb(self.image_root / src_image),
            "trg_img": _open_rgb(self.image_root / trg_image),
            "query_ids": [str(q) for q in rows[col["query_id"]]],
            "src_xys": rows[[col["src_x"], col["src_y"]]].to_numpy(dtype=np.float32).reshape(-1, 2),
            "trg_xys": [],
            "scale": [],
        }
        for query_id, raw, scale in zip(item["query_ids"], rows[col["trg_xys"]], rows[col["scale"]]):
            trg_xys = np.asarray(json.loads(raw), dtype=np.float32).reshape(-1, 2)
            scale = float(scale)
            if len(trg_xys) == 0 or not np.isfinite(trg_xys).all():
                raise ValueError(f"Invalid {col['trg_xys']} for {query_id}")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid {col['scale']} for {query_id}")
            item["trg_xys"].append(trg_xys)
            item["scale"].append(scale)
        return item
