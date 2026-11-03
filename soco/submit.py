"""Predict on the test split and write a submission in the SOCO Kaggle format.

sample_submission.csv format (one row per query keypoint, 577 rows):
    id,prediction
    test_q_00341,"[250.0,167.5]"
`prediction` is the [x, y] location in ORIGINAL target-image pixels.

Each dataset sample may hold one keypoint (query-style test file: pair_id == query id)
or several (pair-style: ids become "<pair_id>_<k>" unless `query_ids` gives them explicitly).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .data import build_dataset
from .evaluate import load_model
from .matching import FeatureCache, predict_pair


def fmt_xy(x, y, ndigits=2) -> str:
    return f"[{round(float(x), ndigits)},{round(float(y), ndigits)}]"


def format_rows(sample, pred):
    ids = sample.get("query_ids")
    if ids is None:
        ids = [sample["pair_id"]] if len(pred) == 1 else [f"{sample['pair_id']}_{k}" for k in range(len(pred))]
    return [{"id": i, "prediction": fmt_xy(x, y)} for i, (x, y) in zip(ids, pred)]


def check_against_sample(df: pd.DataFrame, sample_submission: str) -> pd.DataFrame:
    ref = pd.read_csv(sample_submission)
    assert list(ref.columns) == ["id", "prediction"], ref.columns
    missing = set(ref["id"]) - set(df["id"])
    extra = set(df["id"]) - set(ref["id"])
    if extra:
        print(f"WARNING: {len(extra)} predicted ids are not in the sample submission (dropped), e.g. {sorted(extra)[:3]}")
    if missing:
        print(f"WARNING: {len(missing)} ids missing -> filled with the sample's placeholder, e.g. {sorted(missing)[:3]}")
    out = ref[["id"]].merge(df, on="id", how="left")
    out["prediction"] = out["prediction"].fillna(ref["prediction"])
    return out


@torch.no_grad()
def predict_split(cfg, split="test", out_csv="submission.csv", sample_submission: str = ""):
    model, device = load_model(cfg)
    ds = build_dataset(cfg, split=split)  # keep max_pairs=0 for a real submission
    rows, cache = [], FeatureCache()
    for i in tqdm(range(len(ds)), desc="predict"):
        s = ds[i]
        pred, _ = predict_pair(model, s, cfg, device, cache=cache)
        rows += format_rows(s, pred)
    print(f"feature cache: {cache.hits} hits / {cache.misses} misses")
    df = pd.DataFrame(rows)
    if sample_submission:
        df = check_against_sample(df, sample_submission)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} rows -> {out_csv}")
    return df
