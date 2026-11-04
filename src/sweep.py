"""Evaluation sweep: matching settings x feature settings on the validation split (a report only).

Matching settings (flip_test, soft_window, soft_temperature) act after the backbone, so the features
of each (model_id, image_size) are computed once per image pair and reused for every combination.
The result is <run_dir>/sweep.csv. The submission never reads it: copying the best row into BASE
(src/config.py) is a manual, reviewed change (docs/PRD_finetuning.md section 4.3).
"""

import itertools
from copy import deepcopy

import pandas as pd
from tqdm.auto import tqdm

from data.datasets import build_dataset
from src.metrics import PCKMeter
from src.models.matching import choose_source, predict_from_features

MATCHING_KEYS = ("flip_test", "soft_window", "soft_temperature")


def sweep_grid(cfg):
    """[(model_id, image_size)] and [{flip_test, soft_window, soft_temperature}] to evaluate."""
    features = [(m, s) for m in (cfg["sweep_model_id"] or [cfg["model_id"]])
                for s in (cfg["sweep_image_size"] or [cfg["image_size"]])]
    values = [cfg[f"sweep_{k}"] or [cfg[k]] for k in MATCHING_KEYS]
    matching = [dict(zip(MATCHING_KEYS, combo)) for combo in itertools.product(*values)]
    return features, matching


def feature_config(cfg, model_id, image_size):
    fcfg = deepcopy(cfg)
    fcfg["model_id"], fcfg["image_size"] = model_id, image_size
    if model_id != cfg["model_id"]:
        fcfg["model_revision"] = cfg["sweep_model_revisions"].get(model_id) or "main"
        fcfg["head_dim"] = None  # a fresh model: identity head, i.e. its zero-shot features
    return fcfg


def _score_setting(fcfg, model, dataset, matching, cache, verbose):
    """One feature pass over the validation pairs; returns {combo index: PCKMeter}."""
    meters = {i: PCKMeter(fcfg) for i in range(len(matching))}
    need_flip = any(m["flip_test"] for m in matching)
    by_pair = dataset.df.groupby(fcfg["csv_columns"]["pair_id"], sort=False).indices
    for indices in tqdm(list(by_pair.values()), disable=not verbose,
                        desc=f"sweep {fcfg['model_id'].split('/')[-1]} @ {fcfg['image_size']}"):
        samples = [dataset[i] for i in indices]
        first = samples[0]
        src = model.image_features(first["src_img"], cache=cache, key=first["src_image"])
        trg = model.image_features(first["trg_img"], cache=cache, key=first["trg_image"])
        src_flip = model.image_features(first["src_img"], flip=True, cache=cache, key=first["src_image"]) \
            if need_flip else None
        flipped_src, did_flip = choose_source(src, src_flip, trg)  # depends on the images only
        for i, setting in enumerate(matching):
            ccfg = {**fcfg, **setting}
            feats, flipped = (flipped_src, did_flip) if setting["flip_test"] else (src, False)
            for s in samples:
                pred = predict_from_features(feats, trg, s["src_xy"][None], s["src_img"].size, s["trg_img"].size,
                                             ccfg, flipped)[0]
                meters[i].update(s, pred, flipped)
    return meters


def run_sweep(cfg, model, device, cache=None, verbose=True):
    """Return the sweep table (one row per setting, best first), or None when there is nothing to sweep."""
    from src.models import build_model

    features, matching = sweep_grid(cfg)
    if len(features) * len(matching) <= 1:
        return None
    dataset = build_dataset(cfg, "val")
    key = f"pck@{cfg['pck_alpha']}"
    rows = []
    for model_id, image_size in features:
        fcfg = feature_config(cfg, model_id, image_size)
        same_features = (model_id, image_size) == (cfg["model_id"], cfg["image_size"])
        if model_id == cfg["model_id"]:
            sweep_model, saved_cfg = model, model.cfg
            sweep_model.cfg = fcfg  # image_features() reads image_size from model.cfg
        else:
            sweep_model, saved_cfg = build_model(fcfg, device), None
        try:
            meters = _score_setting(fcfg, sweep_model, dataset, matching, cache if same_features else None, verbose)
        finally:
            if saved_cfg is not None:
                model.cfg = saved_cfg
        for i, setting in enumerate(matching):
            summary = meters[i].summary()
            rows.append({"model_id": model_id, "image_size": image_size, **setting,
                         key: round(summary[key], 2), "ci95": round(summary[f"{key}_ci95"], 2),
                         **{k: round(v, 2) for k, v in summary.items()
                            if k.endswith("_pct") or k.startswith(f"{key}/")},
                         "is_current": same_features and all(setting[k] == cfg[k] for k in MATCHING_KEYS)})
        if sweep_model is not model:
            del sweep_model
    return pd.DataFrame(rows).sort_values(key, ascending=False).reset_index(drop=True)


def describe(table, cfg):
    """Short text: the best setting vs the current one (a suggestion, BASE is not changed)."""
    key = f"pck@{cfg['pck_alpha']}"
    best, current = table.iloc[0], table[table["is_current"]]
    cols = ["model_id", "image_size", *MATCHING_KEYS]
    lines = ["Sweep (validation, best first):", table.head(8)[cols + [key, "ci95"]].to_string(index=False)]
    if len(current):
        gain = best[key] - current.iloc[0][key]
        lines.append(f"Best vs current BASE: {gain:+.2f} points. Change BASE by hand only if the gain is "
                     "larger than the interval (ci95) - see docs/PRD_finetuning.md section 6.10.")
    return "\n".join(lines)
