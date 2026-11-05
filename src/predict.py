"""Test predictions and the Kaggle submission file.

predict()          -> <run_dir>/predictions.csv   id,x,y  (original target pixels)
write_submission() -> <run_dir>/submission.csv    id,prediction  with prediction = "[x,y]"

The submission has exactly the ids and order of sample_submission.csv (2 decimals).
It is rejected if an id is missing or repeated, or if a value is not finite.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.datasets import build_dataset
from data.transforms import clip_point
from src.config import data_path


def predict(cfg, model, run_dir):
    """Predict every test query and write predictions.csv (id,x,y)."""
    dataset = build_dataset(cfg, "test")
    cache = model.new_cache()
    model.eval()
    rows = []
    for i in tqdm(range(len(dataset)), desc="predict test"):
        sample = dataset[i]
        pred_xy, _ = model.predict(sample, cfg, cache=cache)
        if cfg["clip_to_image"]:
            pred_xy = clip_point(pred_xy, sample["trg_img"].size)
        rows.append({"id": sample["query_id"], "x": float(pred_xy[0]), "y": float(pred_xy[1])})
    print(f"feature cache: {cache.hits} hits / {cache.misses} misses")
    df = pd.DataFrame(rows, columns=["id", "x", "y"])
    out = Path(run_dir) / "predictions.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} predictions -> {out}")
    return df


def format_xy(x, y, decimals):
    return f"[{round(float(x), decimals)},{round(float(y), decimals)}]"


def write_submission(cfg, predictions, out_csv):
    """predictions: DataFrame id,x,y -> submission CSV aligned with sample_submission.csv."""
    id_col, pred_col = cfg["submission_columns"]
    sample = pd.read_csv(data_path(cfg, "sample_submission_csv"))
    if list(sample.columns) != [id_col, pred_col]:
        raise ValueError(f"sample_submission.csv columns are {list(sample.columns)}, expected {[id_col, pred_col]}")

    preds = predictions.copy()
    preds["id"] = preds["id"].astype(str)
    if preds["id"].duplicated().any():
        raise ValueError(f"Repeated ids in predictions: {preds.loc[preds['id'].duplicated(), 'id'].head(3).tolist()}")
    if not np.isfinite(preds[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    expected = sample[id_col].astype(str)
    missing = sorted(set(expected) - set(preds["id"]))
    extra = sorted(set(preds["id"]) - set(expected))
    if missing:
        raise ValueError(f"{len(missing)} ids of sample_submission.csv have no prediction, e.g. {missing[:3]}")
    if extra:
        print(f"Warning: {len(extra)} predicted ids are not in sample_submission.csv (dropped), e.g. {extra[:3]}")

    preds = preds.set_index("id").loc[expected]
    submission = pd.DataFrame({
        id_col: expected.to_numpy(),
        pred_col: [format_xy(x, y, cfg["prediction_decimals"]) for x, y in zip(preds["x"], preds["y"])],
    })
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_csv, index=False)
    print(f"wrote {len(submission)} rows -> {out_csv}")
    return submission


def check_submission(cfg, submission_csv):
    """Re-read a submission file and check it against sample_submission.csv. Returns a report dict."""
    id_col, pred_col = cfg["submission_columns"]
    sub = pd.read_csv(submission_csv)
    sample = pd.read_csv(data_path(cfg, "sample_submission_csv"))
    points = np.array([json.loads(p) for p in sub[pred_col]], dtype=float)
    return {
        "columns_ok": list(sub.columns) == [id_col, pred_col],
        "same_ids_and_order": sub[id_col].astype(str).tolist() == sample[id_col].astype(str).tolist(),
        "unique_ids": bool(sub[id_col].is_unique),
        "all_finite": bool(np.isfinite(points).all()),
        "n_rows": len(sub),
    }
