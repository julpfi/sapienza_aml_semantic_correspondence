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

## D-009 (2026-11-19): DINOv3 ViT-B/16 as an additional backbone

- **Decision:** add `src/models/backbone_dinov3.py` (`Dinov3Backbone`, a subclass of `DinoBackbone` with the same
  interface) and the preset `DINOV3_B16` in `src/config.py`: `facebook/dinov3-vitb16-pretrain-lvd1689m` at a pinned
  revision, `image_size` 736, `aug_edge_margin` 16. `build_backbone` chooses the class from `model_id`; `BASE` stays
  on DINOv2.
- **Why:** after fine-tuning, about 18 points of queries lie between PCK@0.05 and PCK@0.10 (precision), and DINOv3
  is trained for cleaner dense features at high resolution (Gram anchoring, RoPE).
- **Choices:** B/16 at 736 px gives the 46 × 46 grid of DINOv2 at 644, so the comparison measures the features,
  not the grid size, and every setting counted in cells is unchanged. RoPE coordinate augmentation stays off
  (the module is kept in eval mode). Larger models and resolutions only after the frozen comparison.
- **Rejected:** a new `backbone` key (the model id already says which family it is, and old runs need no
  migration); editing `DinoBackbone` (the DINOv2 path must stay byte-for-byte the same).

## D-010 (2026-11-20): first DINOv3 run through `ACTIVE_PRESET`

- **Decision:** `src/config.py` gets `DINOV3_FIRST_RUN` (= `DINOV3_B16` + `fusion` "none" + a DINOv3 evaluation
  sweep: `soft_window` 7/9/11/13, `soft_temperature` 0.01/0.02/0.04, `image_size` 736/896/1024) and `ACTIVE_PRESET`,
  which `get_config()` applies on top of `BASE` for new runs. `BASE` itself stays on DINOv2.
- **Why:** switching the backbone touches four keys at once; one line on and off is easier to undo than editing
  `BASE` by hand (D-005 rule: undo the edit before the next run). Saved runs load their own `config.json` filled
  from `BASE`, so the preset never changes how an old run is evaluated or resumed.
- **Hyperparameters:** the D-008 training of `BASE` unchanged (same 46 × 46 grid and token count, so cell-based
  settings, memory and micro-batch carry over); no SD fusion, so the run measures the backbone alone.
- **Sweep:** the existing evaluate-only sweep, not a W&B training sweep: one run's interval is about ±4.5 points,
  so training settings are decided by two-seed runs and `compare_runs.py`, as in experiments-sd.md.

## D-011 (2026-11-20): DINOv3 fine-tuning at the best frozen sweep setting (user request)

- **Decision:** `ACTIVE_PRESET = DINOV3_FT_896`: DINOv3-B/16 at 896 px (56 × 56 cells), `flip_test` True,
  `soft_window` 7, `soft_temperature` 0.02, `fusion` "none"; training keys as in `BASE` (D-008).
- **Why:** frozen sweep of run `20261120-104658_evaluate`: 35.5 val PCK@0.10 at this setting vs 24.7 at 736 / no
  flip / window 11 (frozen DINOv2: 41.1). Every top setting used `flip_test`, a sign of the positional (left/right)
  bias reported for raw DINOv3 features (INSID3, CVPR 2026).
- **Risk:** the matching settings differ from every DINOv2 run, so comparisons need `compare_runs.py --allow` or a
  re-evaluation at equal settings; the window was chosen on frozen features and is re-swept after training.
