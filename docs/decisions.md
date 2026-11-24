# Decisions

One entry per non-trivial decision. Newest last. Never contradict an entry silently: add a new one.
Earlier entries referenced in the code: D-004 = no test suite (checks are compileall + the submit dry-run),
D-005 = one configuration (`BASE` in `src/config.py`), no command-line overrides of settings.

## D-007 (2026-11-16): v3 additions on branch `sd`, all off by default

- **Decision:** add pair-batched training, hard negatives from the pair's other keypoints, LoRA, EMA and
  Stable Diffusion feature fusion as `BASE` keys whose defaults reproduce v2 (docs/PLAN.md).
- **Why:** the errors left after fine-tuning are near misses and wrong-keypoint (left/right) mix-ups; these
  changes target them, and each can be tested alone against v2 with a paired comparison.
- **Rejected:** a left/right keypoint swap table (the data has no keypoint ids or categories); `peft`
  (hand-written LoRA avoids a dependency); precomputed SD features (517 train images make live SD affordable
  and keep the existing augmentation); gradient clipping at 1 (clips every update at this loss temperature).
- **Resources approved by the user:** `diffusers==0.43.0`, read-only Hub/PyPI lookups, a tiny SD test pipeline
  for CPU checks, subagents.
- **SD weights:** `stabilityai/stable-diffusion-2-1` is not public any more (checked 2026-11-16); the mirror
  `sd2-community/stable-diffusion-2-1` is pinned by commit in `BASE`.

## D-008 (2026-11-17): BASE = recommended stage-1 setting + `fusion` "concat" (user request)

- **Decision:** `BASE` no longer reproduces v2. It holds: `loss` marginal, `pair_batch` True
  (`pair_effective_batch` 4), `epochs` 8, `warmup_epochs` 1.5, `loss_temperature_epochs` 3.0, `hard_neg_weight` 3.0,
  `ema_decay` 0.99, `train_mode` finetune, `label_jitter` 0.25, `soft_window` 11 / `soft_temperature` 0.02,
  lighter augmentation (crop scale [0.8, 1.0], rotation 10, color jitter 0.2 at p 0.5, hue 0.02, grayscale 0.05,
  blur 0.05), `lr_adapter` 3e-4, and `fusion` "concat" (DINOv2 fine-tuned on `cat(dino, 0.5 * frozen SD)`).
- **Why:** the user asked to train with SD fusion and the recommended hyperparameters directly.
  Pair mode gives ~78 updates per epoch instead of ~127, so 8 epochs, 1.5 warmup epochs and a 3-epoch temperature
  ramp keep the v2 totals in updates. Lighter augmentation and label jitter: small objects got worse in run 1.
- **Not applied (stage 2 of the recommendation):** `lr_backbone` 5e-6, `epochs` 4 / `warmup_epochs` 0.5,
  `loss_temperature_start` None. They assume `init_artifact` = a trained stage-1 model; from raw DINOv2 they would
  under-train. Use them when fine-tuning a trained model with `fusion` "adapter".
- **v2:** `V2_TRAINING` in `src/config.py` holds the v2 values of every changed training, augmentation and fusion
  key; `BASE` updated with it is R0. The v2-equivalence checks use it. Old runs still load through `V3_OFF` and their
  own `config.json`.
- **Risk:** this run mixes several changes; its effect cannot be split per change without the R1-R4 ablations.

## D-009 (2026-11-24): Stable Diffusion fusion for DINOv2-large as the preset `unified_lora_large_sd`

- **Decision:** add the preset `unified_lora_large_sd` = `UNIFIED_LORA_LARGE` (DINOv2-L/14, LoRA on the last 6
  blocks) + `fusion` "adapter" + multi-layer SD features (`sd_up_blocks` [0, 1, 2] of SD 2.1, each block
  L2-normalised and resized to the 46-cell DINOv2 grid, then concatenated) + `sd_noise_samples` 4 (noise ensemble
  at evaluation only; training uses one sample) + `adapter_dropout` 0.2 (Dropout2d on the SD input of the adapter).
  The three new keys are in `BASE` with values that keep the current behaviour (None / 1 / 0.0).
  Also fixed: `--fusion` and `--cross-transform` only apply to new training runs (they changed the architecture of
  saved runs in `--run`, `--resume` and `--ensemble`); `load_model_weights` warns about missing trainable keys
  (an adapter loaded without its weights started from random init without a message); the Colab notebook passes
  `--fusion` only when `FUSION` is set, so it no longer overrides the preset with "none".
- **Why:** the competition rules require a diffusion model (slides, "The rules"). The literature note
  `docs/literature/2026-11-24-sd-dinov2-large-fusion.md` supports multi-layer SD features with per-part L2
  normalisation (SD+DINO), a noise ensemble (DIFT: about -1.6 points without one) and feature dropout against
  overfitting (GeoAware-SC; Jamais Vu shows supervised models overfit seen keypoints). The adapter starts at zero,
  so update 0 is the DINOv2-L LoRA model and the SD part can only add what training finds useful.
- **Rejected:** `concat` on DINOv2-L (about 4200 channels per cell with three SD blocks, no learned mixing);
  input sizes 840 / 960 (DINOv2-L LoRA training next to the SD UNet on a T4; left as an experiment); PCA of the SD
  features (the adapter's 1x1 input convolution does the reduction); a class-name prompt (the data has no
  categories); removing the presets and command-line flags.
- **Conflict to resolve:** the presets (`PRESETS`, `SC_PRESET`, `--preset`) and the command-line overrides added in
  commits 76e1d19 and 8c468af contradict D-005 (one configuration, no override flags) and were never logged here.
  This entry builds on them because the team's Colab notebook uses them; the user has to confirm whether D-005 is
  replaced. `BASE["fusion"]` is "none" on disk although D-008 set "concat"; the on-disk value is kept.
- **Unknown:** no paper reports DINOv2-L with SD. The gain must be measured on Colab: `unified_lora_large_sd`
  against `unified_lora_large`, seeds 0 and 1, with `scripts/compare_runs.py`.

## D-010 (2026-11-24): presets and a few command-line flags replace D-005; `BASE["fusion"]` stays "none" (user decision)

- **Decision:** D-005 is replaced. `BASE` in `src/config.py` stays the default configuration. Named presets
  (`PRESETS`, chosen with `--preset`, `SC_PRESET` or `ACTIVE_PRESET`) and the command-line flags of `run.py` are
  allowed. Every run still saves its full resolved configuration to `config.json`. Architecture flags
  (`--fusion`, `--cross-transform`) only apply to new training runs (D-009). `BASE["fusion"]` stays "none": the
  fusion part of D-008 ("concat") is superseded. Stable Diffusion fusion is used through the preset
  `unified_lora_large_sd` (D-009).
- **Why:** the team's Colab notebook already selects experiments by preset (commits 76e1d19, 8c468af), and the
  user confirmed this way of working on 2026-11-24.
- **Rejected:** keeping D-005 and treating the presets as a violation to remove; restoring `BASE["fusion"]` to
  "concat", which would load Stable Diffusion in every `BASE` run.
- **Checks:** `scripts/check_lora_ema.py` and `scripts/check_sd_fusion.py` now switch off the features they do not
  test (`cross_transform`, `lora_train_layernorm`, `object_zoom`) in their own test configuration, because `BASE`
  turned `cross_transform` on. Their assertions are unchanged.

## D-011 (2026-11-24): fusion mode "adapter_mix", a learned merge of DINOv2-L and the SD adapter (off by default)

- **Decision:** add `fusion: "adapter_mix"`: `fused = normalize(mix(cat(dino, adapter(sd))))`, where `mix` is a
  1x1 convolution from 2 x 1024 to 1024 channels initialised with weight `[I | I]` and bias 0. The SD adapter is the
  same as in "adapter" (its last convolution starts at zero), so at update 0 the output equals the "adapter" mode
  and the DINOv2-L LoRA model. New preset `unified_lora_large_sd_mix` = `unified_lora_large_sd` + `fusion`
  "adapter_mix". `BASE["fusion"]` stays "none"; no new `BASE` key.
- **Why:** the user wants the combination of the two branches to be learned during fine-tuning instead of a fixed
  sum. The `[I | I]` initialisation keeps the starting point of D-009, so a paired comparison measures only the
  learned merge. With `[I | 0]` the SD branch would receive no gradient at all (its last layer starts at zero).
- **Rejected:** a trained DINOv2 head (redundant with LoRA; the old head at lr 1e-3 flattened the similarity maps);
  a per-channel gate only (cheaper, but it cannot mix channels across the two branches); a randomly initialised mix
  convolution (the first epochs would start from a worse model than DINOv2-L).
- **Order:** first train `unified_lora_large_sd` as the baseline (seeds 0 and 1), then `unified_lora_large_sd_mix`
  with the same seeds, and decide with `scripts/compare_runs.py`.
