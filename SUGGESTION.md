# Suggestions (cheapest first)

Context: run 1 = 62.9 val PCK@0.10 (66.0 with the 11×11 / T 0.02 window). Details in
[docs/REVIEW_finetuning.md](docs/REVIEW_finetuning.md); settings and charts in [docs/RUN.md](docs/RUN.md) §8–9.

1. **Free +3 points now.** In run 1's `config.json` set `soft_window: 11`, `soft_temperature: 0.02`, then
   `evaluate --run` (expect ≈ 66.0) and `submit --run`. No GPU training needed.
2. **Run 2 with the new `BASE`** (1 block, AdamW + OneCycleLR, head 1e-3 / backbone 1e-5, clip 5, ~10%
   augmentation). Check: entropy not far above 3.5, flat `embed_var_*` (a head at 1e-3 can flatten the maps),
   `val/pck@0.1/single` not falling. If val PCK still rises at epoch 4, go back to 6–7 epochs.
3. **If run 2 is worse than ~62 at the same window:** change one thing at a time: `lr_head` 1e-4, then
   `grad_clip` 20 (5 clips almost every update, so the gradient size no longer varies between updates), then
   `unfreeze_blocks` 2.
4. **Resolution 756 / 840** with the fine-tuned model: near misses (16%) and small objects are the largest
   remaining error, and a finer grid fixes them directly.
5. **Left/right confusions** (`wrong_keypoint` ≈ 10%, unchanged by fine-tuning): use the other queries' targets
   of the same pair as hard negatives in the loss, or test `flip_test` again on the run 2 model.
6. **LP-FT** if the entropy overshoot persists: 1 epoch with the head only, then unfreeze block 11.
7. **Noise level:** repeat the best config with a second seed, so a gain can be told apart from run-to-run noise.
8. **Repo housekeeping:** `.claude/` is gitignored, so the agents and the cleanup skill are not shared; the skill
   is `.claude/skills/repo_cleanup.md`, but Claude Code only loads `.claude/skills/repo-cleanup/SKILL.md`.
   The `.claude/` agent and skill files point to docs that do not exist (`docs/decisions.md`,
   `docs/project-rules.md`, `docs/REORGANIZATION.md`).
   `data/*.py` and `data/README.md` still mention "Lang": the project permissions block edits under `data/`.
