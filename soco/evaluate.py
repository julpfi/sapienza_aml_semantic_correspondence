"""Stage 1: training-free DINO evaluation, plus a cached sweep over cheap matching options."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .backbone import DinoFeatures
from .data import build_dataset
from .matching import FeatureCache, alignment_score, extract, predict_from_features
from .metrics import PCKMeter


def load_model(cfg):
    device = cfg.resolved_device()
    return DinoFeatures(cfg).to(device).eval(), device


@torch.no_grad()
def evaluate(cfg, model=None, device=None, ds=None, save=True, verbose=True):
    if model is None:
        model, device = load_model(cfg)
    ds = ds or build_dataset(cfg)
    meter = PCKMeter(cfg.pck_alphas, cfg.pck_norm)
    cache = FeatureCache()
    t0 = time.time()
    for i in tqdm(range(len(ds)), disable=not verbose, desc=cfg.run_name):
        s = ds[i]
        fs = extract(model, s["src_img"], cfg, device, cache=cache, key=s.get("src_path"))
        ft = extract(model, s["trg_img"], cfg, device, cache=cache, key=s.get("trg_path"))
        flipped = False
        if cfg.flip_test:
            fsf = extract(model, s["src_img"], cfg, device, flip=True, cache=cache, key=s.get("src_path"))
            if alignment_score(fsf, ft) > alignment_score(fs, ft):
                fs, flipped = fsf, True
        pred = predict_from_features(fs, ft, s["src_kps"], s["src_img"].size, s["trg_img"].size, cfg, flipped)
        meter.update(s, pred, flipped)
    summ = meter.summary()
    summ["sec_per_pair"] = (time.time() - t0) / max(len(ds), 1)
    if save:
        out = Path(cfg.out_dir) / cfg.run_name
        out.mkdir(parents=True, exist_ok=True)
        cfg.save(out / "config.json")
        (out / "summary.json").write_text(json.dumps(summ, indent=2))
        meter.per_category().to_csv(out / "per_category.csv")
        meter.frame().to_csv(out / "per_keypoint.csv", index=False)
    if verbose:
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in summ.items()}, indent=2))
        print(meter.per_category().to_string())
    return summ, meter


@torch.no_grad()
def sweep_matching(cfg, grid: dict, ds=None):
    """Sweep options that do NOT change the features (matcher, window, temperature, upsample, flip).
    Features are extracted once per image, so this costs roughly one evaluation."""
    model, device = load_model(cfg)
    ds = ds or build_dataset(cfg)
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    meters = [PCKMeter(cfg.pck_alphas, cfg.pck_norm) for _ in combos]
    cfgs = [cfg.replace(**c) for c in combos]
    cache = FeatureCache()
    for i in tqdm(range(len(ds)), desc="sweep"):
        s = ds[i]
        fs = extract(model, s["src_img"], cfg, device, cache=cache, key=s.get("src_path"))
        ft = extract(model, s["trg_img"], cfg, device, cache=cache, key=s.get("trg_path"))
        need_flip = any(c.flip_test for c in cfgs)
        fsf = extract(model, s["src_img"], cfg, device, flip=True, cache=cache, key=s.get("src_path")) if need_flip else None
        use_flip = need_flip and alignment_score(fsf, ft) > alignment_score(fs, ft)
        for c, m in zip(cfgs, meters):
            fl = c.flip_test and use_flip
            pred = predict_from_features(fsf if fl else fs, ft, s["src_kps"], s["src_img"].size,
                                         s["trg_img"].size, c, fl)
            m.update(s, pred, fl)
    rows = [{**combo, **m.summary()} for combo, m in zip(combos, meters)]
    df = pd.DataFrame(rows).sort_values("pck@0.1 (per-kp)", ascending=False)
    out = Path(cfg.out_dir) / f"{cfg.run_name}_sweep"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep.csv", index=False)
    print(df.round(2).to_string(index=False))
    return df


def sweep_features(cfg, grid: dict, ds=None):
    """Sweep options that DO change features (backbone, layer, img_size). One full eval per combo."""
    ds = ds or build_dataset(cfg)
    rows = []
    for v in itertools.product(*grid.values()):
        combo = dict(zip(grid, v))
        name = cfg.run_name + "_" + "_".join(f"{k}{val}" for k, val in combo.items())
        c = cfg.replace(run_name=name, **combo)
        summ, _ = evaluate(c, ds=ds, verbose=False)
        rows.append({**combo, **summ})
        print(combo, f"PCK@0.1={summ['pck@0.1 (per-kp)']:.2f}")
    df = pd.DataFrame(rows).sort_values("pck@0.1 (per-kp)", ascending=False)
    out = Path(cfg.out_dir) / f"{cfg.run_name}_featsweep"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep.csv", index=False)
    return df
