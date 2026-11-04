## Suggested next steps (cheapest first)

1. Log the LB score of this submission next to val 40.4.
2. Resolution sweep above 644 (756, 840, 952; must be a multiple of 14), and try `dinov2_vitl14`.
3. Sweep `flip_test` and `soft_window` size on the competition validation set.
4. Fix `finetune` for `AMLQueryDataset` (issue 1) and run Stage 2 (2 blocks, 2 epochs) on GPU.
5. Then Stage 3 (see `.claude/pipeline.md`): geometry-aware matching for left/right confusions, or
   SD+DINO fusion.
