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


def size_group(scale, bins):
    """Object size group from L: "small" < bins[0] <= "medium" < bins[1] <= "large"."""
    return "small" if scale < bins[0] else "medium" if scale < bins[1] else "large"


def edge_distance(trg_xys, wh, scale):
    """How far the most interior accepted target is from the image border, divided by L."""
    trg_xys = np.asarray(trg_xys, dtype=np.float64).reshape(-1, 2)
    w, h = wh
    border = np.minimum.reduce([trg_xys[:, 0], trg_xys[:, 1], w - trg_xys[:, 0], h - trg_xys[:, 1]])
    return float(border.max() / scale)


def confidence_interval(pck_percent, n):
    """Half width of the 95% normal-approximation interval, in PCK points."""
    p = pck_percent / 100
    return float(196 * np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))


class PCKMeter:
    """Collects one row per query and reports PCK at several alphas (in percent).

    Each row also has the groups and the error type used by the evaluation report
    (docs/PRD_finetuning.md section 4.2): size group, single/multi target, edge distance, and
    near miss / wrong keypoint / far miss for wrong predictions.
    """

    def __init__(self, cfg):
        self.alphas = list(cfg["pck_alphas_report"])
        if cfg["pck_alpha"] not in self.alphas:
            self.alphas.append(cfg["pck_alpha"])
        self.main_alpha = cfg["pck_alpha"]
        self.strict = cfg["pck_strict"]
        self.cfg = cfg
        self.rows = []
        self.targets = {}  # query_id -> accepted targets [N, 2], for the wrong-keypoint check

    def update(self, sample, pred_xy, flipped=False):
        if sample.get("trg_xys") is None:
            return
        cfg = self.cfg
        self.targets[sample["query_id"]] = np.asarray(sample["trg_xys"], dtype=np.float64).reshape(-1, 2)
        self.rows.append({
            "query_id": sample["query_id"],
            "pair_id": sample["pair_id"],
            "category": sample["category"],
            "pred_x": float(pred_xy[0]),
            "pred_y": float(pred_xy[1]),
            "dist": normalized_distance(pred_xy, sample["trg_xys"], sample["scale"]),
            "flipped": flipped,
            "scale": float(sample["scale"]),
            "size_group": size_group(sample["scale"], cfg["train_group_bins"]),
            "n_accepted": len(self.targets[sample["query_id"]]),
            "target_group": "single" if len(self.targets[sample["query_id"]]) == 1 else "multi",
            "trg_edge_dist": edge_distance(sample["trg_xys"], sample["trg_img"].size, sample["scale"]),
        })

    def _error_types(self, df):
        """near_miss | wrong_keypoint (landed on another query's target of the same pair) | far_miss | other."""
        alpha, cfg = self.main_alpha, self.cfg
        types = []
        pair_queries = df.groupby("pair_id")["query_id"].apply(list).to_dict()
        for row in df.itertuples():
            if is_correct(row.dist, alpha, self.strict):
                types.append("correct")
                continue
            pred = np.array([row.pred_x, row.pred_y])
            others = [self.targets[q] for q in pair_queries[row.pair_id] if q != row.query_id]
            if others and min(np.linalg.norm(t - pred, axis=1).min() for t in others) / row.scale < alpha:
                types.append("wrong_keypoint")
            elif row.dist < cfg["near_miss_factor"] * alpha:
                types.append("near_miss")
            elif row.dist > cfg["far_miss_dist"]:
                types.append("far_miss")
            else:
                types.append("other_miss")
        return types

    def frame(self):
        cols = ["query_id", "pair_id", "category", "pred_x", "pred_y", "dist", "flipped", "scale",
                "size_group", "n_accepted", "target_group", "trg_edge_dist"]
        df = pd.DataFrame(self.rows, columns=cols)
        for alpha in self.alphas:
            df[f"pck@{alpha}"] = is_correct(df["dist"], alpha, self.strict).astype(float)
        bins = self.cfg["edge_bins"]
        df["edge_group"] = pd.cut(df["trg_edge_dist"], [-np.inf, bins[0], bins[1], np.inf],
                                  labels=[f"<{bins[0]}", f"{bins[0]}-{bins[1]}", f">{bins[1]}"], right=False).astype(str)
        df["error_type"] = self._error_types(df) if len(df) else []
        return df

    def summary(self):
        df = self.frame()
        out = {"n_queries": len(df), "n_pairs": int(df["pair_id"].nunique())}
        for alpha in self.alphas:
            out[f"pck@{alpha}"] = 100 * df[f"pck@{alpha}"].mean() if len(df) else float("nan")
        key = f"pck@{self.main_alpha}"
        out[f"{key}_ci95"] = confidence_interval(out[key], len(df))
        out["score"] = out[key] / 100  # competition score, 0-1
        out["flip_rate"] = 100 * df["flipped"].mean() if len(df) else 0.0
        for kind, pct in (100 * df["error_type"].value_counts(normalize=True)).items():
            out[f"{kind}_pct"] = float(pct)
        for column in ("size_group", "target_group"):
            for group, value in (100 * df.groupby(column)[key].mean()).items():
                out[f"{key}/{group}"] = float(value)
        return out

    def per_group(self):
        """PCK, query count, 95% interval and error mix for each size / target / edge group."""
        df = self.frame()
        key = f"pck@{self.main_alpha}"
        rows = []
        for column in ("size_group", "target_group", "edge_group"):
            for group, part in df.groupby(column):
                pck = 100 * part[key].mean()
                rows.append({"grouping": column, "group": group, "n_queries": len(part), key: round(pck, 1),
                             "ci95": round(confidence_interval(pck, len(part)), 1),
                             "median_dist": round(float(part["dist"].median()), 3),
                             **{f"{k}_pct": round(100 * v, 1)
                                for k, v in part["error_type"].value_counts(normalize=True).items()}})
        return pd.DataFrame(rows)

    def error_types(self):
        df = self.frame()
        counts = df["error_type"].value_counts()
        return pd.DataFrame({"n_queries": counts, "pct": (100 * counts / max(len(df), 1)).round(1)})


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
