"""PCK@alpha, per keypoint and per category."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def pck_threshold(sample, norm: str) -> float:
    if norm == "bbox" and sample.get("trg_bbox") is not None:
        x1, y1, x2, y2 = sample["trg_bbox"]
        return float(max(x2 - x1, y2 - y1))
    w, h = sample["trg_img"].size
    return float(max(w, h))


class PCKMeter:
    def __init__(self, alphas, norm="bbox"):
        self.alphas, self.norm = tuple(alphas), norm
        self.rows = []   # one row per keypoint

    def update(self, sample, pred: np.ndarray, flipped: bool = False):
        gt = sample["trg_kps"]
        if gt is None:
            return
        d = np.linalg.norm(pred - gt, axis=-1) / pck_threshold(sample, self.norm)
        for i, di in enumerate(d):
            self.rows.append({"pair_id": sample["pair_id"], "category": sample["category"],
                              "kp": i, "dist": float(di), "flipped": flipped})

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        for a in self.alphas:
            df[f"pck@{a}"] = (df["dist"] <= a).astype(float)
        return df

    def summary(self) -> dict:
        df = self.frame()
        cols = [f"pck@{a}" for a in self.alphas]
        out = {"n_pairs": df["pair_id"].nunique(), "n_kps": len(df)}
        for c in cols:
            out[f"{c} (per-kp)"] = 100 * df[c].mean()
            out[f"{c} (per-img)"] = 100 * df.groupby("pair_id")[c].mean().mean()
        out["flip_rate"] = 100 * df.groupby("pair_id")["flipped"].first().mean()
        return out

    def per_category(self) -> pd.DataFrame:
        df = self.frame()
        cols = [f"pck@{a}" for a in self.alphas]
        return (100 * df.groupby("category")[cols].mean()).round(1).sort_values(cols[1] if len(cols) > 1 else cols[0])
