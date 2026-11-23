"""Validation splits:
1. Legacy pair split (`splits/val_pair_ids.json`, when cfg["val_fold"] is None):
   Same algorithm as the Lang notebook (shuffle unique pair ids with seed 42, keep first 20%).
2. Image-disjoint K-fold split (`splits/val_disjoint_folds.json`, when cfg["val_fold"] is 0..K-1):
   Partitions connected components of the image-pair graph across K folds so train and val share
   ZERO images (matching the 0%-overlap test set).
"""

from collections import defaultdict
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


def make_disjoint_folds(train_df, cfg, n_folds=None):
    """Assign connected components of (source_image, target_image) to n_folds image-disjoint folds,
    balancing query count, multi-target share, and small-object share across folds.
    Returns dict {str(fold): [pair_id, ...]} with zero image overlap between any two folds."""
    n_folds = int(n_folds or cfg.get("val_n_folds", 5))
    col = cfg["csv_columns"]
    pair_col, src_col, trg_col = col["pair_id"], col["src_image"], col["trg_image"]
    trg_xys_col, scale_col = col["trg_xys"], col["scale"]

    pairs = train_df[[pair_col, src_col, trg_col]].drop_duplicates()
    has_meta = trg_xys_col in train_df.columns and scale_col in train_df.columns
    is_multi = (train_df[trg_xys_col].apply(lambda s: len(json.loads(s)) > 1)
                if has_meta else None)
    is_small = (train_df[scale_col] < 150.0) if has_meta else None

    adj = defaultdict(set)
    for _, r in pairs.iterrows():
        s, t = str(r[src_col]), str(r[trg_col])
        adj[s].add(t)
        adj[t].add(s)

    visited, comps = set(), []
    src_str = train_df[src_col].astype(str)
    for img in sorted(adj):
        if img in visited:
            continue
        stack, comp = [img], set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            stack.extend(adj[u] - visited)
        mask = src_str.isin(comp)
        comp_pairs = sorted(str(p) for p in train_df.loc[mask, pair_col].unique())
        n_q = int(mask.sum())
        n_m = int(is_multi[mask].sum()) if has_meta else 0
        n_s = int(is_small[mask].sum()) if has_meta else 0
        comps.append((n_q, n_m, n_s, comp_pairs, comp))

    # Largest components first (deterministic tie-break by first pair_id), minimizing total
    # normalized squared imbalance in (queries, multi-target queries, small-object queries).
    comps.sort(key=lambda x: (-x[0], -len(x[3]), x[3][0]))
    tot_q = max(sum(x[0] for x in comps) / n_folds, 1.0)
    tot_m = max(sum(x[1] for x in comps) / n_folds, 1.0)
    tot_s = max(sum(x[2] for x in comps) / n_folds, 1.0)

    fold_pairs = {str(f): [] for f in range(n_folds)}
    fold_imgs = {str(f): set() for f in range(n_folds)}
    fq, fm, fs = [0] * n_folds, [0] * n_folds, [0] * n_folds
    for n_q, n_m, n_s, comp_pairs, comp_imgs in comps:
        def _delta(f):
            old = ((fq[f] - tot_q) / tot_q) ** 2
            new = ((fq[f] + n_q - tot_q) / tot_q) ** 2
            if has_meta:
                old += ((fm[f] - tot_m) / tot_m) ** 2 + ((fs[f] - tot_s) / tot_s) ** 2
                new += ((fm[f] + n_m - tot_m) / tot_m) ** 2 + ((fs[f] + n_s - tot_s) / tot_s) ** 2
            return (new - old, len(fold_pairs[str(f)]), f)

        best_f = min(range(n_folds), key=_delta)
        fold_pairs[str(best_f)].extend(comp_pairs)
        fold_imgs[str(best_f)].update(comp_imgs)
        fq[best_f] += n_q
        fm[best_f] += n_m
        fs[best_f] += n_s

    # Verify strict 0-image overlap between all pairs of folds.
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            overlap = fold_imgs[str(i)] & fold_imgs[str(j)]
            if overlap:
                raise RuntimeError(f"Disjoint fold bug: folds {i} and {j} share {len(overlap)} images.")
    return fold_pairs


def split_path(cfg):
    return Path(cfg["splits_dir"]) / cfg["val_split_file"]


def disjoint_folds_path(cfg):
    return Path(cfg["splits_dir"]) / cfg.get("val_folds_file", "val_disjoint_folds.json")


def load_disjoint_folds(train_df, cfg):
    """Read splits/val_disjoint_folds.json, or build and save it on first use."""
    path = disjoint_folds_path(cfg)
    known = set(train_df[cfg["csv_columns"]["pair_id"]].astype(str))
    if path.exists():
        saved = json.loads(path.read_text())
        folds = saved["folds"]
        all_ids = [str(p) for ids in folds.values() for p in ids]
        unknown = [p for p in all_ids if p not in known]
        if unknown:
            raise ValueError(f"{path} has pair ids not in train.csv, e.g. {unknown[:3]}. Delete to rebuild.")
        return folds
    n_folds = int(cfg.get("val_n_folds", 5))
    folds = make_disjoint_folds(train_df, cfg, n_folds=n_folds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "n_folds": n_folds,
        "image_disjoint": True,
        "folds": folds,
    }, indent=1) + "\n")
    print(f"Saved {n_folds} image-disjoint folds to {path}. Commit this file.")
    return folds


def load_val_pair_ids(train_df, cfg):
    """Read the validation pair ids: image-disjoint fold when cfg['val_fold'] is set, else legacy split."""
    val_fold = cfg.get("val_fold")
    if val_fold is not None:
        folds = load_disjoint_folds(train_df, cfg)
        key = str(int(val_fold))
        if key not in folds:
            raise ValueError(f"val_fold {val_fold} not in {sorted(folds)}.")
        return folds[key]

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
