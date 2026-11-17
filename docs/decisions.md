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
