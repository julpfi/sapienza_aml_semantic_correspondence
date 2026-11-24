# Review: D-011 fusion mode adapter_mix (2026-11-24)

- **Brief:** quality review of the uncommitted D-011 change (fusion mode `adapter_mix`, preset `unified_lora_large_sd_mix`). Only the adapter_mix parts were reviewed; the D-009/D-010 parts are covered in the sections below. Mode: fix.
- **Files reviewed:** src/models/fusion.py, src/config.py, src/train.py, scripts/check_sd_fusion.py, colab/run.ipynb (comments only)
- **Base:** branch sd, commit 4981c2f (changes were uncommitted)

## Verified
- Init: `mix` weight is `[I | I]`, bias 0, so `mix(cat(dino, s)) == dino + s`; with `out` zero, fused equals `normalize(dino)` and equals the `adapter` output. Checked by scripts/check_sd_fusion.py.
- Gradients: at step 1 `out` gets a gradient through the identity s half of `mix`; the s half of `mix.weight` has zero gradient only because `s = 0`, and receives one from step 2. Expected, and covered by a check.
- Checkpoints: `fusion.mix.weight` and `fusion.mix.bias` are parameters with `requires_grad`, so `trainable_state` saves them and `load_model_weights` loads them or warns when missing.
- `configure_trainable` groups `fusion` for both `adapter` and `adapter_mix`. No other place in src/, run.py, scripts/ or data/ compares the fusion kind to `"adapter"` (only `fusion != "none"` / `== "none"` checks, which are correct for the new mode).
- Docstrings (fusion.py, train.py lines 14 and 64), the BASE `fusion` comment, the preset and the Colab comments match the code. No disallowed comments found.

## Modifications
| File:line | Change | Why |
|---|---|---|
| scripts/check_sd_fusion.py (adapter_mix checks) | removed proxy variable `mix_dino = dino.clone()` and use `dino` directly | variable only renamed a copy that was used once; `mix(...)` does not modify its input |
| scripts/check_sd_fusion.py (module names check) | `{k for k in adapter_ref.state_dict()}` -> `set(adapter_ref.state_dict())` | redundant comprehension |

## Not changed (needs CEO decision)
- None. Naming is consistent (`adapter_mix`, `unified_lora_large_sd_mix`, `fusion.mix`). The BASE comment line for `adapter_mix` is long (about 120 characters) but left as is.

## Skipped (do-not-touch list or unsure)
- No BASE key or default touched; nothing in the do-not-touch list was edited.

## Checks
- compileall: ok
- dry-run: ok
- scripts/check_sd_fusion.py: 0 failures; check_lora_ema.py and check_pair_batch.py: ALL PASS

## How to undo
`git checkout 4981c2f -- scripts/check_sd_fusion.py` reverts the two edits (it also reverts the earlier uncommitted check changes, so prefer reverting by hand).

---

# Review: D-009 SD multi-block fusion for DINOv2-large (2026-11-24)

- **Brief:** quality review of the uncommitted D-009 change (multi up-block SD features, noise ensemble, adapter dropout, preset `unified_lora_large_sd`, CLI architecture flags for new runs only, missing-trainable-key warning, Colab FUSION default None). Mode: fix.
- **Files reviewed:** src/models/sd.py, src/models/fusion.py, src/models/__init__.py, src/config.py, scripts/check_sd_fusion.py, run.py, src/utils/checkpoint.py, colab/run.ipynb, docs/RUN.md, docs/decisions.md
- **Base:** branch sd, commit 4981c2f (changes were uncommitted)

## Modifications
| File:line | Change | Why |
|---|---|---|
| src/models/sd.py (module docstring) | documented `sd_up_blocks` and the noise ensemble | the docstring still described only one block |
| src/models/sd.py (class docstring) | signature now lists `n_samples` | the forward signature changed |
| src/models/sd.py (hook registration) | `max(self.up_blocks)` computed once as `deepest_needed`, with a WHY comment (blocks run 0, 1, 2, ...) | was recomputed per loop iteration; the order is the non-obvious part |
| src/models/sd.py (`_make_hook`) | removed the inline comment on `if stop:` | said what the code does, now covered by `deepest_needed` |

## Verified without change
- Hooks sit on every listed block; the hook of `max(sd_up_blocks)` raises `_StopForward`, and diffusers runs `up_blocks` in the order 0, 1, 2, 3, so every needed block is captured before the stop.
- Per-block resize to `out_hw`, per-block L2 norm only when `sd_up_blocks` is set, concat, final L2 norm. With `sd_up_blocks=None` the result is `normalize(resize(feats))` as before, so the single-block path is unchanged (the check script asserts `torch.equal` for `n_samples=1`).
- The noise ensemble sums raw float32 features and divides by `n_samples` before normalisation. Sample 0 uses seed `seed + 0`, the same noise as before; the noise is rebuilt from a CPU `torch.Generator`, so it is deterministic and stays float32.
- Hooks are registered once in `__init__` and write to `self._captured`, which is reset after each sample; they do not accumulate. `forward` is under `@torch.no_grad()`; the UNet runs in fp16 on CUDA and features are cast to float32 before use.
- `out_dim = sum(reversed(block_out_channels)[b])`; for SD 2.1 and blocks [0, 1, 2] this is 1280 + 1280 + 640 = 3200, which is also the `Fusion` in-conv input size (`Fusion` gets `sd.out_dim`).
- `Dropout2d` has no parameters, so state-dict keys are unchanged; it is active only in train mode, and the zero-initialised output layer still gives fused == dino at init (checked in the script, eval mode).
- `CorrespondenceModel.features` passes 1 sample when `self.training`, else `sd_noise_samples`; zoom, flip and ensemble paths all go through `features`, so they get the evaluation ensemble.
- run.py: `new_run` is false for `--resume`, `--run` and `--ensemble` (also for ensemble members), so `--fusion` and `--cross-transform` only change new runs and a warning is printed when they would differ. Checkpoint warning lists the first 5 missing trainable names and does not raise.
- colab/run.ipynb parses with `json.load`; `--fusion` is passed only when `FUSION` is set.
- docs/RUN.md, docs/decisions.md and `UNIFIED_LORA_LARGE_SD` agree on key names and values (`sd_up_blocks` [0, 1, 2], `sd_noise_samples` 4, `adapter_dropout` 0.2, fusion "adapter", model facebook/dinov2-large). `V3_OFF` has the three new keys, so old runs load with the old behaviour.
- The cell counts in the docs (23 / 46 / 92 at 736) were not measured here; the check script uses a tiny model. Confirm on a Colab dry run.

## Not changed (needs CEO decision)
- docs/decisions.md D-009 already flags the conflict with D-005 (presets and CLI overrides). Nothing to add.
- A duplicate index such as `sd_up_blocks=[1, 1]` is accepted and gives the channels twice (consistent with `out_dim`, but pointless). No validation was added.
- On an exception other than `_StopForward`, `_captured` keeps tensors until the next call. Harmless; not changed.

## Skipped (do-not-touch list or unsure)
- Pre-existing failures, unrelated to this diff (they also fail at HEAD): check_sd_fusion "none: state-dict keys equal the v2 (e5cedb8) model's" and "checkpoint: only fusion. keys"; check_lora_ema "4a saved state holds only lora keys" (not run here).
- The long comment on `BASE["sd_up_blocks"]` is allowed (where the value comes from and what it accepts).

## Checks
- compileall: ok
- dry-run (`run.py --mode submit --dry-run`): ok ("Dry run OK: all needed files are present.")
- check_sd_fusion.py: only the 2 known failures; all new D-009 checks pass
- check_pair_batch.py: ALL PASS
- colab/run.ipynb json.load: ok

## How to undo
`git diff 4981c2f -- src/models/sd.py`; `git checkout 4981c2f -- src/models/sd.py` reverts my edits together with the D-009 sd.py change, so use `git diff` and revert by hand to keep D-009.

## Addendum (2026-11-24): stale check configurations

- `scripts/check_lora_ema.py` and `scripts/check_sd_fusion.py`: `make_cfg` now sets `cross_transform`,
  `lora_train_layernorm` and `object_zoom` to False. The checks built their models from `BASE`, which has
  `cross_transform` True, so trainable `cross.*` parameters appeared in the saved state and three checks failed
  (also at HEAD). No assertion changed. Result: `check_lora_ema.py` ALL PASS, `check_sd_fusion.py` 0 failure(s).
- Not covered by any check: the model with `cross_transform` or `lora_train_layernorm` on.

## Addendum (2026-11-24): CPU smoke run of the preset path and a scheduler fix

- Smoke run (driver in the session scratchpad, no repo file used for it): `unified_lora_large_sd` with DINOv2-base
  instead of large (large is not downloaded locally), the tiny SD test pipeline (`sd_up_blocks` [0, 1]), 224 px,
  16 train / 16 validation queries, 1 epoch, CPU. Four optimiser steps ran. `best.pt` holds 72 LoRA and 16 `fusion.`
  keys and no SD keys; it reloads without a missing-key warning. SD was called with one noise sample in training
  steps and with `sd_noise_samples` (2) in every evaluation pass; object zoom stage 2 ran. The PCK of this run has
  no meaning. Not exercised: the 1024-d DINOv2-L features and the real SD 2.1 block sizes (1280 / 1280 / 640).
- `src/train.py`: `warmup_updates` is capped at `total_updates - 1`. With `warmup_epochs >= epochs`, `pct_start` was
  1.0 and the last `OneCycleLR` step raised ZeroDivisionError (found by the smoke run with `epochs` 1). Runs with
  fewer warmup updates than total updates are unchanged.
- Real SD 2.1 shapes (UNet built from the pinned `unet/config.json` on the meta device, no weights, 736 px input,
  timestep 261): `up_blocks` 0 / 1 / 2 give 1280x23x23 / 1280x46x46 / 640x92x92, 3200 channels in total. This
  matches `SDFeatures.out_dim` and the adapter input of `unified_lora_large_sd`.
