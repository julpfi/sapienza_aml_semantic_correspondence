"""Validation: predictions + PCK on the held-out image pairs, with the full report.

Outputs in <run_dir>/ (skipped with save=False, as during training):
    summary.json      PCK@alphas (+95% interval), score (0-1), error mix, PCK per group, val loss
    per_query.csv     one row per query: prediction, d/L, size/target/edge group, error type
    per_group.csv     PCK, count, interval and error mix per size / single-multi target / edge group
    error_types.csv   correct, near_miss, wrong_keypoint, far_miss, other_miss counts
Groups and error types: docs/PRD_finetuning.md section 4.2.
"""

import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from data.datasets import build_dataset, read_queries
from data.transforms import to_resized
from src.metrics import PCKMeter, official_score
from src.models.loss import matching_loss


def validation_loss(cfg, model, sample, cache):
    """Training loss of one validation query, from the cached features of predict()."""
    src_feats = model.image_features(sample["src_img"], cache=cache, key=sample["src_image"])
    trg_feats = model.image_features(sample["trg_img"], cache=cache, key=sample["trg_image"])
    size = cfg["image_size"]
    loss, _ = matching_loss(src_feats.float(), trg_feats.float(),
                            to_resized(sample["src_xy"][None], sample["src_img"].size, size),
                            to_resized(sample["trg_xys"], sample["trg_img"].size, size),
                            size, cfg["loss_temperature"], cfg["label_sigma"])
    return float(loss)


def evaluate(cfg, model, run_dir, save=True, verbose=True, with_loss=True, cache=None):
    """Predict every validation query and score it. Returns (summary, meter)."""
    dataset = build_dataset(cfg, "val")
    meter = PCKMeter(cfg)
    cache = cache if cache is not None else model.new_cache()
    model.eval()
    losses = {}
    start = time.time()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), disable=not verbose, desc="evaluate val"):
            sample = dataset[i]
            pred_xy, flipped = model.predict(sample, cfg, cache=cache)
            meter.update(sample, pred_xy, flipped)
            if with_loss:
                losses[sample["query_id"]] = validation_loss(cfg, model, sample, cache)
    summary = meter.summary()
    summary["sec_per_query"] = (time.time() - start) / max(len(dataset), 1)

    frame = meter.frame()
    if losses:
        frame["loss"] = frame["query_id"].map(losses)
        summary["loss"] = float(frame["loss"].mean())
        for column in ("size_group", "target_group"):
            for group, value in frame.groupby(column)["loss"].mean().items():
                summary[f"loss/{group}"] = float(value)

    official = official_score(cfg, read_queries(cfg, "val"), frame)
    if official is not None:
        summary["official_score"] = official

    if save:
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        frame.to_csv(out / "per_query.csv", index=False)
        meter.per_group().to_csv(out / "per_group.csv", index=False)
        meter.error_types().to_csv(out / "error_types.csv")
    if verbose:
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}, indent=2))
    return summary, meter
