"""Decision gate: read validation results and recommend the next stage.

The thresholds are deliberately explicit so you can change them. Reference points
(SPair-71k, PCK@0.1, bbox-normalized): zero-shot DINOv2 nearest-neighbour is ~55;
supervised DINOv2-based methods reach ~85 in recent papers. Your numbers on the
competition data may differ; calibrate `target` against the leaderboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def recommend(run_dir: str, target: float = 70.0, sanity_floor: float = 35.0,
              baseline_dir: str | None = None, stage: int = 1) -> str:
    run = Path(run_dir)
    s = json.loads((run / "summary.json").read_text())
    cats = pd.read_csv(run / "per_category.csv", index_col=0)
    pck = s["pck@0.1 (per-kp)"]
    lines = [f"Run: {run.name}   PCK@0.1 (per-kp) = {pck:.1f}   target = {target:.1f}"]
    weakest = cats["pck@0.1"].nsmallest(4)
    lines.append("Weakest categories: " + ", ".join(f"{c} {v:.1f}" for c, v in weakest.items()))

    if pck < sanity_floor:
        lines += ["DECISION: STOP AND DEBUG (score is below what frozen DINOv2 should give).",
                  "  - visualise a few predictions (notebook cell 'Visual check')",
                  "  - verify keypoint order is (x, y) and coordinates are rescaled back to original size",
                  "  - verify PCK normalisation (bbox vs image) matches the competition metric"]
        return "\n".join(lines)

    if baseline_dir:
        b = json.loads((Path(baseline_dir) / "summary.json").read_text())["pck@0.1 (per-kp)"]
        lines.append(f"Gain over baseline {Path(baseline_dir).name}: {pck - b:+.1f}")

    if pck >= target:
        lines.append("DECISION: SUBMIT this configuration; spend remaining time on ensembling / TTA.")
        return "\n".join(lines)

    if stage == 1:
        lines += ["DECISION: STAGE 2 - fine-tune the last DINO blocks on the training pairs.",
                  "  First make sure the matching sweep (matcher / flip_test / layer / img_size) is done;",
                  "  carry its best settings into fine-tuning.",
                  "  `python run.py finetune --set ft_unfreeze_blocks=2 ft_epochs=2`"]
    else:
        lines += ["DECISION: STAGE 3 - fine-tuning plateaued below target. Options, cheapest first:",
                  "  a) more blocks / LoRA, larger img_size (e.g. 644 or 840 for patch-14)",
                  "  b) geometry-aware matching: pose/flip alignment + spatial smoothness (optimal transport)",
                  "  c) feature fusion with Stable Diffusion features (SD+DINO) - strongest but heaviest",
                  "  Pick (b) if weakest categories are symmetric objects (left/right confusion);",
                  "  pick (c) if weak categories are texture-poor (plants, TVs, bottles)."]
    return "\n".join(lines)
