"""Validation split: hold out whole image pairs from train.csv.

Same algorithm as the Lang notebook, so validation scores stay comparable with its 40.4:
shuffle the unique pair ids with numpy seed 42 and keep the first 20% for validation.
The chosen ids are saved to splits/val_pair_ids.json (tracked in git) the first time.
"""

import json
from pathlib import Path

import numpy as np


def make_val_pair_ids(train_df, cfg):
    """Return the validation pair ids, in the Lang notebook's order."""
    col = cfg["csv_columns"]["pair_id"]
    pair_ids = train_df[col].drop_duplicates().to_numpy(copy=True)  # writable (pandas 3 copy-on-write)
    if len(pair_ids) < 2:
        raise ValueError("Need at least two image pairs to make a validation split.")
    rng = np.random.default_rng(cfg["split_seed"])
    rng.shuffle(pair_ids)
    n_val = min(len(pair_ids) - 1, max(1, int(cfg["val_fraction"] * len(pair_ids))))
    return [p.item() if hasattr(p, "item") else p for p in pair_ids[:n_val]]


def split_path(cfg):
    return Path(cfg["splits_dir"]) / cfg["val_split_file"]


def load_val_pair_ids(train_df, cfg):
    """Read the saved split, or create and save it. Checks it still matches train.csv."""
    path = split_path(cfg)
    if path.exists():
        saved = json.loads(path.read_text())
        ids = saved["val_pair_ids"]
        known = set(train_df[cfg["csv_columns"]["pair_id"]].astype(str))
        unknown = [i for i in ids if str(i) not in known]
        if unknown:
            raise ValueError(f"{path} has pair ids that are not in train.csv, e.g. {unknown[:3]}. "
                             "Delete the file to rebuild the split.")
        return ids
    ids = make_val_pair_ids(train_df, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "split_seed": cfg["split_seed"],
        "val_fraction": cfg["val_fraction"],
        "n_pairs": len(ids),
        "val_pair_ids": ids,
    }, indent=1) + "\n")
    print(f"Saved a new validation split ({len(ids)} pairs) to {path}. Commit this file.")
    return ids
