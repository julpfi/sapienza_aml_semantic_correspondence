# Suggestions (cheapest first)

Context: run 1 = 62.9 val PCK@0.10 (66.0 with the 11×11 / T 0.02 window). Details in
[docs/REVIEW_finetuning.md](docs/REVIEW_finetuning.md); settings and charts in [docs/RUN.md](docs/RUN.md) §8–9.

1. **Free +3 points now.** In run 1's `config.json` set `soft_window: 11`, `soft_temperature: 0.02`, then
   `evaluate --run` (expect ≈ 66.0) and `submit --run`. No GPU training needed.
2. **Run 2 with the new `BASE`** (1 block, half learning rates, 20% warmup, clip 20, ~10% augmentation).
   Check: smaller entropy peak, `grad_clipped` ≈ 0 after the warmup, flat `embed_var_*`, small objects ≥ 60.
   If val PCK still rises at epoch 4, go back to 6–7 epochs (not to a higher learning rate).
3. **If run 2 is worse than ~62 at the same window:** change one thing at a time: `unfreeze_blocks` 2, then the
   learning rates, then the augmentation. Run 2 changes many settings at once.
4. **Resolution 756 / 840** with the fine-tuned model: near misses (16%) and small objects are the largest
   remaining error, and a finer grid fixes them directly.
5. **Left/right confusions** (`wrong_keypoint` ≈ 10%, unchanged by fine-tuning): use the other queries' targets
   of the same pair as hard negatives in the loss, or test `flip_test` again on the run 2 model.
6. **LP-FT** if the entropy overshoot persists: 1 epoch with the head only, then unfreeze block 11.
7. **Noise level:** repeat the best config with a second seed, so a gain can be told apart from run-to-run noise.
8. **Repo housekeeping:** `.claude/` is gitignored, so the agents and the cleanup skill are not shared; the skill
   is in `.claude/repo_cleanup/` and is only loaded from `.claude/skills/repo-cleanup/`. Several docs point to
   files that do not exist (`docs/decisions.md`, `docs/project-rules.md`, `docs/PRD_reorganization.md`).
