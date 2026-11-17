# Plan: orientation-aware training + LoRA/EMA + SD fusion on branch `sd`

## Context

v2 (DINOv2-B/14, last 2 blocks fine-tuned, soft-argmax window) reached val PCK@0.10 ≈ 63-66. The errors left are
near misses (16%) and wrong-keypoint / left-right mix-ups (10%). The TASK asks for supervision changes that target
orientation without SD first, then a cheap frozen SD test, then SD fusion only if that test passes.

Your answers: build **steps 1-3** (no zoom-in / ensemble), write a **new small compare script**, SD is **allowed**
by the competition.

Phase 0 findings (read-only):
- Branch `sd` exists, equals `v2` at `e5cedb8`, working tree clean. The Plan A code is only on `clip`.
- There is **no test suite** (pytest is installed, no tests). The baseline checks are
  `python -m compileall -q run.py src data scripts` and `python run.py --mode submit --dry-run`.
- No local GPU (torch CPU build). **All training and the SD gate run on Colab, by you.** Agents test on CPU only.
- "Measure first" (TASK order 1) already exists: `src/metrics.py` reports `pck@0.05` and `wrong_keypoint_pct` in
  `summary.json` and W&B. No code needed, only the compare script.
- `.claude/settings.json` denies edits under `data/**`, so all new code goes in `src/` and `scripts/`.
- `diffusers` and `peft` are not installed. LoRA will be hand-written (about 60 lines), so only `diffusers` is new.
- Training is per query today (`batch_loss` in `src/train.py`, one backbone pass per query image pair);
  augmentation is per query (`augment_sample` in `src/augment.py`).

On approval, step 1 is copying this plan to `docs/PLAN.md` (plan mode allows only this file now).

## Design (every feature is a `BASE` key, default off, so `sd` with defaults reproduces v2 exactly)

> Superseded for the defaults by D-008 (2026-11-17, user request): `BASE` now holds the recommended stage-1 setting
> plus `fusion` "concat"; v2 = `BASE` + `V2_TRAINING`, and the v2-equivalence checks apply `V2_TRAINING`.

| Feature | Key(s) in `BASE` (`src/config.py`) | How |
|---|---|---|
| Marginal loss | `loss: "marginal"` (exists) | no code; a run setting |
| Pair-batched training | `pair_batch: False`, `pair_effective_batch: 4` | one item = one pair with all its queries; one backbone pass supervises ≈ 6 queries |
| Hard negatives | `hard_neg_weight: 1.0` (1 = off), `hard_neg_radius: 1.0`, `hard_neg_exclusion: 2.0` (cells) | cells near **other queries' accepted targets of the same pair**, minus cells near the query's own accepted targets, get `+log(weight)` on their logit before the softmax. Needs `pair_batch` |
| LoRA | `train_mode: "lora"`, `lora_rank: 8`, `lora_alpha: 16`, `lora_dropout: 0.05`, `lora_blocks: 6`, `lr_lora: 1e-4` | wraps the query/value linears of the last N blocks; B starts at zero, so update 0 = frozen DINOv2 |
| EMA | `ema_decay: 0.0` (0 = off; try 0.99) | average of the trainable weights; validation and `best.pt` use the averaged weights |
| SD features | `fusion: "none" \| "concat" \| "adapter"`, `sd_model_id`, `sd_model_revision`, `sd_image_size: 736`, `sd_timestep: 261`, `sd_up_block: 1`, `sd_concat_weight: 0.5` | frozen SD UNet + VAE, fp16 on GPU, one noising step with a fixed seed, features from decoder up-block 1, empty-prompt embedding computed once. Bilinear resize to the DINOv2 grid if the shapes differ |
| Fusion adapter | `adapter_layers: 2`, `adapter_dim: 256`, `lr_adapter: 1e-4` | `fused = normalize(dino + adapter(sd))`, last adapter layer zero-initialised, so update 0 = the DINOv2 model. `concat` (no training) is the frozen gate |

Unchanged: matching, soft-argmax window, `src/evaluate.py`, `src/predict.py`, metric, split, submission, `data/`.
Grad clip stays at the current value (10). New keys are read with `cfg.get(key, default)` so old runs still load.

## Tasks (no file overlap)

| Task | Goal | Owns | Acceptance check (CPU) |
|---|---|---|---|
| T0 lead | `BASE` keys, `docs/PLAN.md`, `docs/decisions.md` | `src/config.py`, `docs/PLAN.md`, `docs/decisions.md` | compileall + dry-run pass |
| T1 trainer | Pair dataset, pair augmentation, pair loss with hard negatives, training-loop wiring (also the hooks for T2/T3: LoRA and adapter parameter groups, EMA update / swap) | `src/train.py`, `src/augment.py`, `src/models/loss.py`, `src/pair_batch.py` (new), `scripts/check_pair_batch.py` (new) | with `hard_neg_weight 1` the pair loss equals the mean of the existing per-query `matching_loss` (1e-5); hard-negative mask is correct on a synthetic pair; all keys off → the first 3 batches and their loss match HEAD `src/train.py` |
| T2 lora-ema | LoRA module + EMA helper | `src/models/lora.py` (new), `src/utils/ema.py` (new), `scripts/check_lora_ema.py` (new) | LoRA at init gives features equal to frozen (1e-6); only LoRA params trainable; checkpoint save → rebuild → load round-trip; EMA math on a toy model |
| T3 sd-fusion | SD feature extractor, concat and adapter fusion, model wiring (calls `apply_lora`) | `src/models/sd.py` (new), `src/models/fusion.py` (new), `src/models/__init__.py`, `scripts/check_sd_fusion.py` (new) | on a tiny SD test pipeline: feature shape, determinism (same seed → same features), adapter at init gives `fused == dino` (1e-6), `fusion "none"` path unchanged, flip cache keys still work |
| T4 tools-docs | Paired compare script, run book, dependency pin, Colab notes | `scripts/compare_runs.py` (new), `docs/experiments-sd.md` (new), `docs/RUN.md`, `environment.yml`, `colab/run.ipynb` | `compare_runs.py` on two fake `per_query.csv` gives the expected difference, interval and refuses mismatched query ids or matching settings |

Reuse, do not rewrite: `read_queries` / `QueryDataset` (`data/datasets.py`), `flip_sample`, `to_resized`,
`preprocess_image` (`data/transforms.py`), `_augment_image` / `_geometric` (`src/augment.py`),
`positive_cells`, `target_distribution`, `matching_loss` (`src/models/loss.py`), `trainable_state` /
`load_model_weights` (`src/utils/checkpoint.py`, already saves any `requires_grad` parameter, so LoRA and
adapter weights need no change there), `PCKMeter` error types (`src/metrics.py`), `local_model_dir`
(`src/models/backbone.py`).

Order: T0 → T1, T2, T3, T4 in parallel → T1 wires the T2/T3 hooks into `src/train.py` → verification → review.

## Proposed team (5 agents)

| Role | Count | Model | Owns | Why |
|---|---|---|---|---|
| Lead (this session) | 1 | Fable | T0, briefs, integration decisions, `docs/RESUME.md` | one owner for `src/config.py` and the plan |
| trainer | 1 | Opus | T1 | `src/train.py` is the riskiest file (resume, schedules, equivalence) |
| lora-ema | 1 | Sonnet | T2 | small, well-specified modules |
| sd-fusion | 1 | Opus | T3 | diffusers internals and the identity-at-init guarantees |
| tools-docs | 1 | Sonnet | T4 | independent script and docs |
| reviewer | 1 | Opus, read-only | wrote no code; checks the diff against this plan | TASK Phase 3 |

Agents are general-purpose subagents with self-contained briefs pointing at `docs/PLAN.md`. The existing
`code-reviewer` runs a quality pass after the reviewer. Worktrees are not needed: files do not overlap.

## RESOURCE REQUEST (nothing is installed, downloaded or called until you approve each row)

| # | Resource | Type | Why it is needed | What happens without it |
|---|---|---|---|---|
| 1 | `diffusers` (one pinned version) added to `environment.yml` and pip-installed into the local `advance` env | New dependency + package install | SD UNet / VAE loading and the noising step | No SD code can run; the plan shrinks to T1, T2, T4 |
| 2 | Hugging Face Hub and PyPI lookups (read-only web requests) | Internet access | Confirm which SD 2.1 repo is still downloadable, pin its commit, pick a `diffusers` version compatible with transformers 5.22.1 / torch 2.15 | `sd_model_id` / revision and the pin stay as placeholders you fill in; risk of a first failure on Colab |
| 3 | Download of a tiny SD test pipeline (a few MB) from the Hub | Internet access (external download) | CPU smoke test of the SD extractor and fusion (`check_sd_fusion.py`) | SD code is only compile-checked; its first real execution is your Colab gate |
| 4 | Subagents (4 implementers + 1 reviewer, models as in the team table) | Claude usage | parallel work with separate file ownership | I do the tasks one after another in this session: slower, same result |

Not requested: no API keys (I will not read `.env`; `HF_TOKEN` and `WANDB_API_KEY` are only used by you on Colab),
no MCP server, plugin or skill, no `peft`, no `accelerate`, no full SD download locally (≈ 5 GB, Colab only),
no Kaggle call, no W&B call (smoke runs use `wandb_mode: "disabled"`).

## Risks and hard-to-undo items

- **SD 2.1 availability**: the original repo may be gone; a mirror or SD 1.5 is the fallback (same UNet layout).
- **T4 memory**: SD (fp16) live next to DINOv2 fine-tuning in fp32. The micro-batch probe runs the real forward,
  so it adapts, but batches may get small. Fallback: `fusion` training with DINOv2 LoRA instead of unfrozen blocks.
- **Pair batching changes the update count** (≈ 78 updates per epoch at 4 pairs instead of 127). Runs with and
  without it differ in more than one thing; the run book compares them at their own best epoch.
- **Hard negatives can be label noise** when another query's part is a valid match that is not in the accepted
  list. The exclusion radius and a moderate weight (try 2-4) limit this; the paired wrong-keypoint rate decides.
- **Validation noise** (453 queries): decisions only through `scripts/compare_runs.py`.
- **`src/train.py` smoke edits of `BASE`** are restored to the on-disk values recorded before the edit, never to HEAD.
- Hard to undo: nothing. Work is committed in small steps on `sd`, never pushed or merged. The only change outside
  the repo is the `diffusers` pip install (resource 1), removable with `pip uninstall diffusers`.

## Verification (Phase 3)

1. `python -m compileall -q run.py src data scripts` and `python run.py --mode submit --dry-run`.
2. `python scripts/check_pair_batch.py`, `check_lora_ema.py`, `check_sd_fusion.py`, and `compare_runs.py` on fake
   CSVs: all PASS.
3. v2 equivalence: with every new key at its default, the first batches and their loss match HEAD.
4. Capped CPU smoke train (16 queries, 1 epoch, 224 px, W&B off), once per feature: `pair_batch` + hard negatives +
   marginal; `train_mode "lora"` + EMA; `fusion "adapter"` with the tiny SD pipeline. Purpose: execute the code
   path only. **No real fine-tuning, no uncapped evaluate.**
5. Reviewer agent (wrote no code) checks the diff against `docs/PLAN.md`; then `code-reviewer`.
6. There is no lint tool in the env (ruff missing) and no build step; I will say so in the report, not claim a pass.

Then `docs/RESUME.md` (TASK Phase 4) with the real command outputs, and `docs/experiments-sd.md` with your Colab
run order: B0 (v2) → marginal → pair batch + hard negatives → LoRA + EMA (paired vs the previous best) → frozen SD
`concat` gate (pass: about +4 over frozen DINOv2 at the same settings) → adapter fusion. SD is kept only if the
paired comparison improves PCK@0.05 or the wrong-keypoint rate.

## Status (2026-11-16): all tasks done

| Task | Status | Acceptance check, as run on the final code |
|---|---|---|
| T0 lead | done | `python -m compileall -q run.py src data scripts` exit 0; `python run.py --mode submit --dry-run` → "Dry run OK: all needed files are present." |
| T1 trainer | done | `python scripts/check_pair_batch.py` → 22 PASS, "ALL PASS" (v2 reference pinned to `e5cedb8`) |
| T2 lora-ema | done | `python scripts/check_lora_ema.py` → 22 PASS, "ALL PASS" |
| T3 sd-fusion | done | `python scripts/check_sd_fusion.py` → 24 PASS, "0 failure(s)" (tiny SD test pipeline) |
| T4 tools-docs | done | `python scripts/compare_runs.py --self-test` → 12 PASS, "All checks passed." |
| Verification | done | `python -m pytest tests -q` → 10 passed (the build checks are part of the suite); `python -m ruff check .` → All checks passed!; build (no build system: bytecode compile exit 0, 27 of 27 modules import, `pip check` clean); 5 capped CPU smoke trainings |
| Review | done | independent reviewer: PASS WITH ISSUES; 1 blocker + 4 issues fixed (commit `f9b2324`); quality pass: 15 findings, 9 fixed, 4 known, 2 partly fixed (`docs/RESUME.md` section 7); second independent review of `fe10491`: no blocker, no code issue, 3 report corrections applied; smoke trainings and an EMA resume test re-run on `6ac9463`; third independent review of `c519474`: no blocker, 3 small findings fixed |

Additions outside the ownership table, approved or found during verification: `tests/test_v3.py`, `ruff.toml`,
`src/utils/run.py` (old runs load with the v3 switches off), one unused variable removed in `src/metrics.py`.
Full report: `docs/RESUME.md`.
