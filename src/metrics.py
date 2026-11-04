"""PCK@alpha for the competition (CLAUDE.md, Lang notebook):

    d(q) = min over accepted targets a of ||pred(q) - a||_2
    correct(q) = d(q) / L(q) < alpha        (L = `scale` column, strict "<")

If the official metric.py is in data_dir, official_score() runs it as well, so the two can be
compared. It is loaded read-only and never copied into the repo.
"""

import importlib.util
import json

import numpy as np
import pandas as pd

from src.config import data_path


def normalized_distance(pred_xy, trg_xys, scale):
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(pred_xy).all():
        raise ValueError(f"Prediction is not finite: {pred_xy}")
    trg_xys = np.asarray(trg_xys, dtype=np.float64).reshape(-1, 2)
    return float(np.linalg.norm(trg_xys - pred_xy, axis=1).min() / scale)


def is_correct(dist, alpha, strict=True):
    return dist < alpha if strict else dist <= alpha


class PCKMeter:
    """Collects one row per query and reports PCK at several alphas (in percent)."""

    def __init__(self, cfg):
        self.alphas = list(cfg["pck_alphas_report"])
        if cfg["pck_alpha"] not in self.alphas:
            self.alphas.append(cfg["pck_alpha"])
        self.main_alpha = cfg["pck_alpha"]
        self.strict = cfg["pck_strict"]
        self.rows = []

    def update(self, sample, pred_xy, flipped=False):
        if sample.get("trg_xys") is None:
            return
        self.rows.append({
            "query_id": sample["query_id"],
            "pair_id": sample["pair_id"],
            "category": sample["category"],
            "pred_x": float(pred_xy[0]),
            "pred_y": float(pred_xy[1]),
            "dist": normalized_distance(pred_xy, sample["trg_xys"], sample["scale"]),
            "flipped": flipped,
        })

    def frame(self):
        df = pd.DataFrame(self.rows, columns=["query_id", "pair_id", "category", "pred_x", "pred_y",
                                              "dist", "flipped"])
        for alpha in self.alphas:
            df[f"pck@{alpha}"] = is_correct(df["dist"], alpha, self.strict).astype(float)
        return df

    def summary(self):
        df = self.frame()
        out = {"n_queries": len(df), "n_pairs": int(df["pair_id"].nunique())}
        for alpha in self.alphas:
            out[f"pck@{alpha}"] = 100 * df[f"pck@{alpha}"].mean() if len(df) else float("nan")
        out["score"] = out[f"pck@{self.main_alpha}"] / 100  # competition score, 0-1
        out["flip_rate"] = 100 * df["flipped"].mean() if len(df) else 0.0
        return out

    def per_category(self):
        df = self.frame()
        cols = [f"pck@{a}" for a in self.alphas]
        return (100 * df.groupby("category")[cols].mean()).round(1).sort_values(f"pck@{self.main_alpha}")


def official_score(cfg, solution_df, predictions_df):
    """Score with the official metric.py if it is in data_dir. Returns None when unavailable.

    The official signature is not documented in the repo; the usual Kaggle form
    score(solution, submission, row_id_column_name) is tried. Any failure is reported, not raised.
    """
    path = data_path(cfg, "official_metric_py")
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("official_metric", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        id_col, pred_col = cfg["submission_columns"]
        submission = pd.DataFrame({
            id_col: predictions_df["query_id"],
            pred_col: [json.dumps([round(x, cfg["prediction_decimals"]), round(y, cfg["prediction_decimals"])])
                       for x, y in zip(predictions_df["pred_x"], predictions_df["pred_y"])],
        })
        return float(module.score(solution_df.reset_index(drop=True), submission, id_col))
    except Exception as err:  # report and continue with the local metric
        print(f"Official metric.py could not be used ({type(err).__name__}: {err}).")
        return None
