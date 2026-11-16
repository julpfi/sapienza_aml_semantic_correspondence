# RESUME: v3 additions on branch `sd` (2026-11-16)

Report of the work planned in [PLAN.md](PLAN.md). Base: `e5cedb8` (= `v2`). Nothing was pushed or merged.
Nothing was trained for real: every training result below is a capped CPU smoke run whose numbers mean nothing.

## 1. Summary

The branch `sd` now holds five additions to the v2 pipeline, each behind a `BASE` key that is off by default:
pair-batched training, hard negatives taken from the pair's other keypoints, LoRA, EMA of the trained weights,
and frozen Stable Diffusion features fused with DINOv2 (a no-training `concat` gate and a zero-initialised
adapter). With the defaults, loader, loss, parameter groups, model keys and features are identical to v2 (checked
against `e5cedb8`). A paired run-comparison script, a Colab run book, a pytest wrapper and a lint config were added.
An independent reviewer found one blocker and four issues; all five are fixed. What is left is listed in section 7.
The real test of every addition is the Colab run order in [experiments-sd.md](experiments-sd.md).

## 2. Changes (24 files, +2077 / -58)

| Task | Files |
|---|---|
| T0 lead | `src/config.py` (v3 block of `BASE`, `V3_OFF`), `docs/PLAN.md`, `docs/decisions.md` (D-007) |
| T1 trainer | `src/pair_batch.py` (new: `PairDataset`), `src/augment.py` (`augment_pair`), `src/models/loss.py` (`pair_matching_loss`, `hard_negative_cells`, shared helper), `src/train.py` (pair mode, LoRA / adapter groups, EMA hooks), `scripts/check_pair_batch.py` (new) |
| T2 lora-ema | `src/models/lora.py` (new), `src/utils/ema.py` (new), `scripts/check_lora_ema.py` (new) |
| T3 sd-fusion | `src/models/sd.py` (new: `SDFeatures`), `src/models/fusion.py` (new: `Fusion`), `src/models/__init__.py` (wiring), `scripts/check_sd_fusion.py` (new) |
| T4 tools-docs | `scripts/compare_runs.py` (new), `docs/experiments-sd.md` (new), `docs/RUN.md` (section 8b), `environment.yml` (`diffusers==0.43.0`), `colab/run.ipynb` (compare cell, `BRANCH = "sd"`) |
| Verification | `tests/test_v3.py` (new), `ruff.toml` (new) |
| Outside the plan's file table | `src/utils/run.py` (old runs load with the v3 switches off), `src/metrics.py` (one unused variable removed) |

## 3. Team

| Agent | Model | What it did |
|---|---|---|
| Lead (this session) | Fable | plan, `BASE` keys, lookups, briefs, integration guard, smoke runs, lint, review fixes, commits, this report |
| trainer | Opus | T1 |
| lora-ema | Sonnet | T2 |
| sd-fusion | Opus | T3 |
| tools-docs | Sonnet | T4 |
| reviewer | Opus, read-only | independent review of the whole diff against PLAN.md; wrote no code |
| code-reviewer | Sonnet, report mode | quality pass; changed nothing |

## 4. Resources used

| Resource | Approved | Used for |
|---|---|---|
| `diffusers==0.43.0` | yes | pip-installed into the local `advance` env; pinned in `environment.yml` |
| Hugging Face Hub + PyPI lookups | yes | found that `stabilityai/stable-diffusion-2-1` is not public any more; pinned the mirror `sd2-community/stable-diffusion-2-1` at `bb21548`; chose the diffusers version; read that repo's `unet/config.json` and `vae/config.json` |
| Tiny SD test pipeline (`hf-internal-testing/tiny-stable-diffusion-torch`, pinned) | yes | CPU checks and smoke runs of the SD code |
| Subagents | yes | 4 implementers, 1 reviewer, 1 quality pass |
| `ruff` 0.17.5 | yes (asked during execution) | pip-installed into the local env only; not in `environment.yml` |
| `git fetch origin` | read-only | to confirm that `origin/sd` does not exist |

Not used: no API key, no `.env` read, no MCP server, plugin or skill, no W&B or Kaggle call, no real SD download.

## 5. Decisions and assumptions

- **LoRA is hand-written, no `peft`.** It wraps `q_proj` and `v_proj` (the names in transformers 5.22.1, not
  `query` / `value` as first briefed) of the last `lora_blocks` blocks. In LoRA mode the final LayerNorm is frozen.
- **Hard negatives** are a `+log(weight)` on the logits of cells near the other queries' accepted targets, never
  inside `hard_neg_exclusion` cells of the query's own targets. They need `pair_batch`.
- **Pair mode** backpropagates the sum of the query losses and divides by the update's query count, so an update is
  an exact mean over queries. The effective batch is counted in pairs (`pair_effective_batch`).
- **Pair augmentation never drops a query** (review fix): a crop or rotation that would lose any query is rejected
  and the image keeps its geometry, as in v2.
- **SD recipe:** frozen UNet + VAE, one noising step at `sd_timestep` with one fixed noise tensor (seeded by `seed`),
  features of up-block 1, empty prompt. The SD input is the 644 px DINOv2 image resized to 736 px.
- **Adapter fusion starts as identity** (last layer zero-initialised); `concat` trains nothing and is the gate.
- **`train_mode "frozen"` with fusion is an error**, because that path would skip the SD features.
- **Old runs** (config.json without the v3 keys) load with the switches off (`V3_OFF`), whatever `BASE` says.
- **Tests:** the repo had none (D-004). The suite is a pytest wrapper around the check scripts, as you approved.
- **Lint:** `ruff.toml` selects pycodestyle + pyflakes (what the existing `# noqa: E402` comments refer to) at
  140 columns, the width of the comment columns in `BASE`. Stricter rule sets report style issues in v2 files.
- **Smoke runs used an in-memory config** instead of editing and restoring `BASE`, so `src/config.py` was never
  put in a temporary state.
- **`BRANCH` in `colab/run.ipynb` is now `"sd"`** (it was `"v1"`), so the notebook on this branch runs this code.

## 6. Verification (real outputs, final code)

| Command | Result |
|---|---|
| `python -m pytest tests -q` | `7 passed in 103.20s (0:01:43)` |
| `python -m ruff check .` | `All checks passed!` |
| `python -m compileall -q run.py src data scripts tests` | exit 0 (there is no other build step) |
| `python run.py --mode submit --dry-run` | `Dry run OK: all needed files are present.` |
| `python scripts/check_pair_batch.py` | 22 PASS, `ALL PASS`; v2 equivalence against `e5cedb8` (loss diff 0, same batches); `0 of 670 queries dropped in 200 augmentations` |
| `python scripts/check_lora_ema.py` | 22 PASS, `ALL PASS` |
| `python scripts/check_sd_fusion.py` | 24 PASS, `0 failure(s)` |
| `python scripts/compare_runs.py --self-test` | 12 PASS, `All checks passed.` |
| Old-run loading snippet | with `BASE` set to fusion concat / pair_batch on, an old config loads as `fusion none, pair_batch False, ema_decay 0.0, hard_neg_weight 1.0` |

Capped CPU smoke trainings (16 queries, 1 epoch, 224 px, tiny SD, W&B off): `v2_default`, `pair_hardneg_marginal`,
`lora_ema`, `adapter_fusion`, `all_on`. All five finished, wrote `best.pt`, and a rebuilt model reloaded it and
predicted a finite point. `best.pt` held 24 LoRA tensors in the LoRA runs, 16 adapter tensors in the adapter runs,
and no SD tensor in any run. They ran before the review fixes; the changed code paths (pair augmentation, old-run
loading) were re-checked by the check script and the snippet above, not by a new smoke run.

Independent review: verdict PASS WITH ISSUES. Fixed in `f9b2324`: the blocker (pair augmentation dropped 16% of
the queries) and the four issues (old-run config loading, the equivalence checks compared the code with itself
once committed, `BRANCH`, a wrong example in `compare_runs.py`), plus three nits (`lora_blocks < 1`, two doc lines).

## 7. Not done, known issues, risks

Not done (out of this plan): the zoom-in second pass and the similarity-map ensemble; any real training.

Never run here, first real execution is on Colab:
- the real SD 2.1 weights: loading the pinned mirror, fp16 on the GPU (NaN risk), the expected 46 x 46 x 1280 map
  at 736 px (derived from the UNet config and a scaled-down UNet, not from the real one);
- GPU memory of SD next to DINOv2 fine-tuning, and of the evaluate feature cache with `concat` (2,048 channels);
- a full `train()` in pair, LoRA, EMA or adapter mode at real size; `--resume` with EMA was reviewed by reading only.

Known issues left from the review (not fixed):
- The micro-batch probe uses one-query samples, so pair-mode loss memory is not probed (small next to the backbone).
- The smoke runs used single-query pairs, so no hard negative occurred in them; the check script covers that code.
- With the default radii, few cells can end up as hard negatives when a pair's keypoints are close together
  (`train/hard_neg_cells` near 0): lower `hard_neg_exclusion` before raising the weight.
- A target removed by the edge margin after augmentation is no longer in its query's exclusion zone (rare).
- `sweep_model_id` with fusion on builds a second SD copy on the GPU.
- A `last.pt` without an `"ema"` entry restarts the average from the loaded weights on resume.
- SD features differ slightly between seeds, because the seed sets the SD noise.

Quality findings left as they are (no behaviour change, from the code-reviewer): target parsing and the pair key
are written both in `src/pair_batch.py` and in `data/datasets.py` (`data/` is read-only here); `collate_pairs` is
unused; `hard_negative_cells` recomputes cell distances and is called twice per pair; the v3 keys are read with
`cfg.get(key, default)`, which repeats the `BASE` defaults; `train()` branches on pair mode in about eight places.

Risks: every expected gain in the run book is an estimate; pair mode changes the number of updates per epoch, so
runs with and without it differ in more than one thing; hard negatives can act as label noise.

## 8. Next steps

1. Review the diff: `git diff e5cedb8 sd`. Start with `src/train.py` (`git diff -w`) and `src/models/loss.py`.
2. Push the branch when you are happy with it: `git push -u origin sd` (not done by me).
3. On Colab (`BRANCH = "sd"` is already set), follow [experiments-sd.md](experiments-sd.md) in order:
   R0 baseline → R1 marginal → R2 pair batch → R3 hard negatives → R4 LoRA, EMA → G-SD frozen gate → R5 adapter.
   After each training run, evaluate it (`MODE = "evaluate"`, `RUN = "<run id>"`), then compare with
   `python scripts/compare_runs.py A_RUNS B_RUNS`.
4. The first SD run is the real test of section 7's first list: watch for NaN features and the micro-batch size.

## 9. How to undo

- Everything is on branch `sd`, commits `5c12d30` .. the commit that adds this file, on top of `e5cedb8`.
  `v2`, `clip` and `main` are untouched; nothing was pushed (`origin/sd` does not exist).
- Drop all of it: `git switch v2 && git branch -D sd`.
- Drop one part: `git revert <commit>`; `1c7a4b8` = SD fusion, `db506ab` = pair batching and training hooks,
  `beb671c` = LoRA / EMA modules, `2f1552b` = compare script and docs, `f9b2324` = review fixes.
- Local environment: `pip uninstall diffusers ruff` in the `advance` env. The tiny SD pipeline sits in the
  Hugging Face cache (`~/.cache/huggingface/hub/models--hf-internal-testing--tiny-stable-diffusion-torch`).
