"""Validation: predictions + PCK on the held-out image pairs.

Outputs in <run_dir>/ (skipped with save=False, as during training):
    summary.json      PCK@alphas, score (0-1), counts, seconds per query
    per_query.csv     one row per query: prediction, normalised distance, hits
    per_category.csv  PCK per category ("all" when the CSV has no category)
"""

import json
import time
from pathlib import Path

from tqdm.auto import tqdm

from data.datasets import build_dataset, read_queries
from src.metrics import PCKMeter, official_score


def evaluate(cfg, model, run_dir, save=True, verbose=True):
    """Predict every validation query and score it. Returns (summary, meter)."""
    dataset = build_dataset(cfg, "val")
    meter = PCKMeter(cfg)
    cache = model.new_cache()
    model.eval()
    start = time.time()
    for i in tqdm(range(len(dataset)), disable=not verbose, desc="evaluate val"):
        sample = dataset[i]
        pred_xy, flipped = model.predict(sample, cfg, cache=cache)
        meter.update(sample, pred_xy, flipped)
    summary = meter.summary()
    summary["sec_per_query"] = (time.time() - start) / max(len(dataset), 1)

    official = official_score(cfg, read_queries(cfg, "val"), meter.frame())
    if official is not None:
        summary["official_score"] = official

    if save:
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        meter.frame().to_csv(out / "per_query.csv", index=False)
        meter.per_category().to_csv(out / "per_category.csv")
    if verbose:
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}, indent=2))
    return summary, meter
