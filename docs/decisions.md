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
