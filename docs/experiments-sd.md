# Experiments on branch `sd`: the Colab run book

You run all training on Colab; agents never train. Plan: [PLAN.md](PLAN.md). Decisions: [decisions.md](decisions.md).
Settings and checkpoints: [RUN.md](RUN.md) section 8b.

**Since D-008 ([decisions.md](decisions.md)):** `BASE` is the recommended stage-1 setting with `fusion` "concat",
not v2. The edits in the matrix below are written from v2: for R0, update `BASE` with every key of `V2_TRAINING`
(`src/config.py`); keep `soft_window` 11 / `soft_temperature` 0.02 for all runs. Stage 2 (fine-tune a trained
model with `fusion` "adapter"): `init_artifact` = that model, `lr_backbone` 5e-6, `epochs` 4, `warmup_epochs` 0.5,
`loss_temperature_start` None, `lr_adapter` 3e-4 (try 1e-4 / 1e-3); `train_mode` "lora" if the T4 micro-batch
falls to 1. After the first marginal run, re-run the inference sweep: sharper peaks may prefer a smaller window.

## Rules

1. **One configuration (D-005).** Every run = an edit of `BASE` in `src/config.py`, commit, push to the branch set
   as `BRANCH` in `colab/run.ipynb`, then run. Edit only the keys of the run; undo the edit before the next one.
2. Keep `soft_window`, `soft_temperature`, `flip_test` and `image_size` **fixed** across the whole matrix
   (`compare_runs.py` refuses the comparison otherwise).
3. **Seeds 0 and 1** for every setting (`seed` in `BASE`); two seeds are the unit of a decision. The seed also sets the SD noise, so with `fusion` on the SD features differ slightly between seeds.
4. Decide only with `python scripts/compare_runs.py A_RUNS B_RUNS` (seeds comma-separated, for example
   `r1_s0,r1_s1 r0_s0,r0_s1`). One run's 95% interval is about +-4.5 points on 453 queries, too wide for effects
   of 1-3 points. The paired bootstrap over validation pairs is what counts. In the notebook: set `COMPARE`.
5. **After every training run:** `MODE = "evaluate"`, `RUN = "<run_id>"`. This writes `per_query.csv` into the run
   folder; training alone does not, and `compare_runs.py` needs it.
6. Compare runs of different settings at each run's own best epoch (`best.pt` is the best validation PCK).
7. Fill the results log below as you go.

Known (measured): frozen DINOv2 val PCK@0.10 = 41.1 (window 11) / 40.4 (window 5). Run 1 fine-tuned: 62.9, 66.0
with window 11 / T 0.02. Errors left after run 1: near_miss 16%, wrong_keypoint 10% ([RUN.md](RUN.md),
[REVIEW_finetuning.md](REVIEW_finetuning.md)).

## Run matrix (gated order)

Edits are `key: default -> value` in `BASE`. A run is only started if the previous gate was read.

| Run | Edits to `BASE` (on top of the previous line where stated) | Compare with |
|---|---|---|
| **R0** | none: the v2 baseline | reference |
| **R1** | `loss`: `"soft_ce"` -> `"marginal"` | R0 |
| **R2** | R1 + `pair_batch`: `False` -> `True` | R1 |
| **R3** | R2 + `hard_neg_weight`: `1.0` -> `3.0` (then `2.0` and `4.0` only if R3 helps) | R2 |
| **R4a** | best of R0-R3 + `train_mode`: `"finetune"` -> `"lora"` (the `lora_*` keys and `lr_lora` at defaults) | that best run |
| **R4b** | best of R0-R3 + `ema_decay`: `0.0` -> `0.99` | that best run |
| **R4c** | best of R0-R3 + both (only after reading R4a and R4b separately) | the better of R4a / R4b |
| **G-SD** | no training, see below | frozen DINOv2 |
| **R5** | only if G-SD passes: best trained setting + `fusion`: `"none"` -> `"adapter"` | that best setting |

Notes per run:

- **R2** takes about 78 updates per epoch at `pair_effective_batch` 4 instead of about 127, so runs differ in
  more than one thing. Compare at each run's own best epoch. If R2 is still improving at the last epoch, repeat it
  with more `epochs` (`BASE["epochs"]`) before judging it.
- **R3:** hard negatives can be label noise (another query's part that is a valid match but not in the accepted
  list); the paired wrong_keypoint rate decides, not the overall PCK.
- **R4:** LoRA alone and EMA alone first, then combined.
- **R5 memory (T4):** SD runs in fp16 next to DINOv2. The auto micro-batch adapts to what fits; if it falls to 1,
  use `train_mode` `"lora"` instead of unfrozen blocks.

### G-SD: frozen gate (no training, needs the GPU)

Two evaluations with the same matching settings, no fine-tuning:

1. `fusion` `"none"` (frozen DINOv2): `python run.py --mode evaluate`
2. `fusion` `"concat"` (frozen DINOv2 + SD): edit `BASE`, push, `python run.py --mode evaluate`

The first `concat` run downloads about 5 GB of SD weights (pinned `sd_model_revision`) and needs the GPU. Then
`python scripts/compare_runs.py <concat_run> <none_run>`. **Pass: concat is about +4 points PCK@0.10 or better.**
If it is close, try `sd_concat_weight` 0.3 and 1.0 (default 0.5) and compare each against `none`.

### Keep-rule for SD

Keep SD (R5) only if the paired comparison improves **PCK@0.05 or the wrong_keypoint rate**, not just overall
PCK inside the noise.

## What each run is expected to move (expectations, not results)

| Run | Expected to move |
|---|---|
| R1 marginal | wrong_keypoint rate and multi-target PCK (the model may commit to one target) |
| R2 pair batch | speed and stability of training; PCK little |
| R3 hard negatives | wrong_keypoint rate (left / right and neighbouring parts) and multi-target PCK |
| R4 LoRA / EMA | generalisation: smaller gap between train and val, steadier best epoch |
| R5 SD adapter | far_miss and other_miss rates (semantic confusions DINOv2 cannot resolve) |

## What to watch in W&B

Charts are explained in [RUN.md](RUN.md) section 9.

- `train/loss_over_chance` and `_ema`, `train/entropy_norm`: the first epoch should not spike more than in R0.
- `train/hard_neg_cells`: mean number of hard-negative cells per query (R3 only). Near 0 means the pair's other keypoints
  all fall inside `hard_neg_exclusion`: lower it before raising `hard_neg_weight`.
  Near 0 means the mask does nothing; very large means the radius is too wide.
- `optim/grad_norm`, `optim/grad_clipped`, `batch/effective`, `lr/*` (LoRA and adapter groups appear with their own
  learning rates).
- `val/pck@0.1`, `val/pck@0.05`, `val/pck@0.1/{single, multi}`, `val/pck@0.1_gain_over_ci95`.
- `val/error_types`: wrong_keypoint and near_miss shares.
- `val/entropy_norm` and `val/loss_over_chance` should fall with PCK.

## Results log

Fill in after `evaluate` and `compare_runs.py`. "vs reference" = the paired difference A - B in PCK@0.10 points,
its 95% interval and P(A better), against the run named in the matrix.

| Run | Seed | Edits | Run id | Val PCK@0.10 | PCK@0.05 | wrong_keypoint % | vs reference |
|---|---|---|---|---|---|---|---|
| R0 | 0 | none | | | | | reference |
| R0 | 1 | none | | | | | reference |
| R1 | 0 | `loss` marginal | | | | | |
| R1 | 1 | `loss` marginal | | | | | |
| R2 | 0 | R1 + `pair_batch` | | | | | |
| R2 | 1 | R1 + `pair_batch` | | | | | |
| R3 | 0 | R2 + `hard_neg_weight` 3.0 | | | | | |
| R3 | 1 | R2 + `hard_neg_weight` 3.0 | | | | | |
| R4a | 0 | best + `train_mode` lora | | | | | |
| R4a | 1 | best + `train_mode` lora | | | | | |
| R4b | 0 | best + `ema_decay` 0.99 | | | | | |
| R4b | 1 | best + `ema_decay` 0.99 | | | | | |
| G-SD none | 0 | `fusion` none, evaluate | | | | | reference |
| G-SD concat | 0 | `fusion` concat, evaluate | | | | | |
| R5 | 0 | best + `fusion` adapter | | | | | |
| R5 | 1 | best + `fusion` adapter | | | | | |
