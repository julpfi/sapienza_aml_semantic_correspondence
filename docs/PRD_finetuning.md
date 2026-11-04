# PRD: Fine-tuning DINOv2 for semantic correspondence

Status: **implemented** (v3, see "Implementation status" below) · Date: 2026-11-04 · Branch: `v1` · GPU: **Colab T4 (16 GB)**
Baseline: val PCK@0.10 = 40.4 (frozen ViT-B/14, 644 px, 453 validation queries)

## Implementation status (2026-11-04)

**Data facts, checked on the real download** (they replace the estimates of earlier drafts):
- `train.csv`: 2,495 queries, 392 pairs, 517 images. Validation = 78 pairs = **453 queries**, training split = **2,042 queries**.
- `test.csv`: 577 queries, 98 pairs. 45% of training queries accept more than one target (up to 6).
- Columns: `id, pair_id, source_image, target_image, query_index, source_x, source_y, target_points, scale`.
  **No category or super-category anywhere** (CSV, `*_pairs.json`, image paths). `metric.py` is not in the download.
- Decision (user, 2026-11-04): **super-categories are skipped**. The stratified charts use the groups that the
  data does define: **object size** (L < 100 / 100–300 / > 300 px) and **single vs multiple accepted targets**.
  §4.1 below is kept only as the record of what was considered.

| PRD item | Status | Where |
|---|---|---|
| §3.1 edge-safe soft-argmax window | done | `src/models/matching.py::locate` |
| §3.2 edge-distance groups | done | `src/metrics.py` (`trg_edge_dist`, `edge_group`) |
| §3.3 edge rule for augmentation | done, checked: 0 points moved next to the border in 500 samples | `src/augment.py`, `scripts/check_augmentation.py` |
| §4.2 evaluation report | done (groups by size / target count / edge instead of super-category; error types) | `src/evaluate.py`, `src/metrics.py`, `src/diagnostics.py` |
| §4.3 sweeps | done; matching sweep on by default, feature sweep lists empty (fill them on the T4) | `src/sweep.py` |
| §5.1 `head_dim=None` (identity start, also ViT-L) | done | `src/config.py`, `src/models/__init__.py` |
| §5.2 SDPA attention + kernel check | done | `src/models/backbone.py`, `src/utils/batch_size.py` |
| §5.3 auto micro-batch + effective batch | done | `src/utils/batch_size.py`, `src/train.py` |
| §5.4 batch ramp | done as an option (`effective_batch_schedule`), off by default | `src/train.py` |
| §5.5 augmentation | done | `src/augment.py` |
| §5.6 W&B charts | done (38 keys + tables, histogram, worst-prediction images) | `src/train.py`, `src/diagnostics.py` |
| §6 hyperparameters | in `BASE`; training defaults are now `head="linear"`, `train_mode="finetune"` | `src/config.py` |
| §6.6 early stopping | done (`early_stop_patience`) | `src/train.py` |
| §8 Colab version check | done | `scripts/colab_setup.py` |
| Resume after a disconnect | checked: interrupted + resumed run = identical weights to an uninterrupted run | `src/train.py` |
| Runs on the T4 (§6.11) | **to do by the team** | `colab/run.ipynb` |

Differences from the plan:
- **Swap detection** (§4.2) needs an object axis, and the data has no bounding box. It is replaced by
  **wrong_keypoint**: the prediction lands within 0.10·L of an accepted target of *another* query of the
  same pair (for example the right eye instead of the left). This includes left/right swaps.
- Augmentation lives in `src/augment.py`, not `data/transforms.py` (`.claude/settings.json` blocks edits under `data/`).
- Per-epoch validation also logs the validation **loss**, and the loss per group.

## 0. TL;DR

| # | What | Why | Cost |
|---|---|---|---|
| 1 | Fix the **edge bias** of the soft-argmax window; augmentations **keep targets away from the borders** | Free accuracy; needed before rotation/crop | small |
| 2 | **Evaluation report**: per object size / target count / edge distance, error types, W&B, **sweeps** (`flip_test` × `soft_window`; `image_size` 644→952; ViT-B vs ViT-L) | Know where points are lost before spending T4 hours | medium |
| 3 | **Training on Colab T4**: automatic batch size, augmentation, W&B per super-category, the hyperparameter plan of §6 | The fine-tuning itself | large |
| 4 | **Submission unchanged** | Works and is checked | none |
| 5 | **Colab stays on pip**, not Miniconda (§8) | Faster, keeps Colab's CUDA torch | none |

**Hyperparameter decisions in one line each** (details and evidence in §6):
- **Train:** the last **2** blocks + final LayerNorm + identity-initialised linear head. Then try 1 and 3 blocks (ViT-L: 4).
- **Learning rate:** backbone **1e-5** (try 5e-6 and 2e-5), head **10×** the backbone. Layer-wise decay **0.75** when ≥ 3 blocks are trained.
- **Schedule:** **linear warmup (~10% of steps, ≥ 30 steps) + cosine decay to 1% of the peak**, updated **every step**, not every epoch.
- **Length and batch:** effective batch **16**, **6–8 epochs** with the best epoch kept (early stopping on val PCK). A batch-size ramp is not worth it on this data size.
- **Precision:** fp16 AMP with GradScaler (the T4 has no bf16). Attention through PyTorch SDPA (memory-efficient kernel).

> Conflict with the cleanup D-005 ("no experiments"): the sweeps come back **only as a report inside
> `--mode evaluate`**. Sweep lists live in `BASE`, results go to the run folder, and nothing in a sweep
> changes `BASE` or the submission. There is still one configuration and no `--set`.

---

## 1. Context

- Pipeline: square resize → DINOv2 patch features → cosine similarity of the source point vs every target
  cell → 2× upsampling + 5×5 windowed soft-argmax → (x, y) in original pixels.
- Training already exists in `src/train.py`:
  - `head="linear"|"mlp"` on frozen features, or `train_mode="finetune"` (head + last `unfreeze_blocks` blocks);
  - soft-label loss over **all accepted targets** (`src/models/loss.py`);
  - AMP, OneCycle, resume from `last.pt`;
  - W&B (loss + lr only).
- Training was never run on the real data.
- The `linear` head starts as the **identity** and fine-tuning starts from the pretrained blocks. So with
  `head="linear"`, `train_mode="finetune"` and *no training*, the result is exactly the baseline. These
  settings can stay in `BASE`. `evaluate`/`submit` change only when a trained `best.pt` is loaded with `--run`.
- Categories: `data/README.md` lists a `category` column ("object category"). It is still marked
  **"to confirm on the real data"**, and there is no super-category column (§4.1).

## 2. Goals and non-goals

**Goals**
- G1: val PCK@0.10 clearly above 40.4. Target ≥ 50 after fine-tuning, on the same 453 validation queries.
  A difference counts only if it is ≥ 3 points or confirmed by a second seed (§6.10).
- G2: every training run explains itself in W&B: loss, lr, batch, per-super-category loss and PCK, GPU memory.
- G3: one `evaluate` run says which classes and error types lose points, and which `flip_test`,
  `soft_window`, `image_size` and model to use.
- G4: the submission path, format and checks stay as they are.

**Non-goals:** SD+DINO fusion, LoRA, geometry-aware matching (Stage 3, only if G1 is not met). No change to
the validation split or the metric.

---

## 3. Phase 1: Targets at the image edge

### 3.1 Bug: soft-argmax window at the border (`src/models/matching.py::locate`)
```python
rows = (row[:, None] + offsets[None]).clamp(0, h - 1)
cols = (col[:, None] + offsets[None]).clamp(0, w - 1)
```
When the best cell is at the border, clamping **repeats the edge cell**. With window 5 and `col=0`, the
columns are `[0, 0, 0, 1, 2]`: the edge cell weighs three times and pulls the prediction to the border.
**Fix:** shift the window so it stays inside the map, or give out-of-range offsets weight 0 instead of
clamping. Only predictions within `window//2` cells of the border change. Measure the change on val.

### 3.2 Measure it
`per_query.csv` gets `trg_edge_dist`: the distance of the nearest accepted target to the image border, / L.
Report PCK per bucket `<0.05`, `0.05–0.15`, `>0.15`.

### 3.3 Augmentations keep targets inside
For every geometric augmentation (§5.4):
- move the points with the **same** transform as the image;
- if the source point or any accepted target lands less than `aug_edge_margin` (1 patch = 14 px in resized
  space) from the border, or on padding, sample again (at most 3 times), then fall back to the original sample;
- accepted targets that left the image are dropped. The sample is kept only if at least one remains.

---

## 4. Phase 2: Evaluation mode (no effect on the submission)

### 4.1 Super-categories (the 4 macro-classes): skipped, the data has no categories (see the status section)
100 categories in 4 super-categories: **Transportation (31), Hand-held Objects (20), Furniture (9), Animals (40)**.
1. `category` column (from `data/README.md`): read it as now.
2. Map it with a **committed file** `data/supercategories.json` (`{"category": "super-category"}`). Build it
   once from the category names in `train.csv`, check the counts 31/20/9/40, and have a person review it.
   An unknown category is an error, not a silent "other".
3. If the column turns out to be missing: get the category from the image path (`images/<split>/<category>/...`), then step 2.
4. Last resort, marked "approximate" in every chart: k-means (k = 4) on the CLS embedding of the target image.

**First action on the real data:** print the columns of `train.csv`/`test.csv`, the number of rows, and the
number of queries per category. Write them into `data/README.md` (replacing "to confirm").

### 4.2 What evaluate reports (run folder **and** W&B)

| Output | Content |
|---|---|
| `summary.json` | PCK@0.05/0.10/0.15 overall + per super-category, with a 95% confidence interval |
| `per_category.csv` | PCK per category, worst first, **with query counts** (few queries = noisy) |
| `per_supercategory.csv` | PCK, median error, near/far/swap % per super-category |
| `per_query.csv` | + `supercategory`, `trg_edge_dist`, `scale`, `n_accepted`, `error_type` |
| `error_types.csv` | counts per error type |
| `visual_check.png` | the **worst** queries of each super-category |
| `sweep.csv` | sweep table (§4.3) |

Error types, and where each one comes from in the pipeline:

| Type | Definition | Caused by | Fixed by |
|---|---|---|---|
| near miss | `0.10 ≤ d/L < 0.20` | localisation | resolution, soft-argmax window/temperature |
| wrong keypoint (includes left/right swaps) | prediction within 0.10·L of an accepted target of another query of the same pair | descriptor symmetry / part confusion | flip handling, fine-tuning with flips |
| far miss | `d/L > 0.5` | descriptor quality (wrong part or background) | fine-tuning, larger model |
| edge | target near the border (§3.2) | window / augmentation | §3 |

Also PCK by **object size L** (small objects are unforgiving) and by edge distance.

### 4.3 Sweeps (report only)
New `BASE` keys. An empty list means skip:
```python
"sweep_flip_test":        [False, True],
"sweep_soft_window":      [3, 5, 7, 9, 11],
"sweep_soft_temperature": [0.02, 0.04],
"sweep_image_size":       [],   # e.g. [644, 756, 840, 952]   (new features: expensive)
"sweep_model_id":         [],   # e.g. ["facebook/dinov2-base", "facebook/dinov2-large"]
```
- **Matching sweep:** the features are computed **once** per image and reused, because these settings only
  act after the backbone. 20 combinations cost about one evaluation.
- **Feature sweep:** one feature pass per combination, on the T4. `dinov2_vitl14` = `facebook/dinov2-large`
  (304M parameters, 24 blocks, 1024-d). Pin its revision like the base model.
- Output: `sweep.csv`, a W&B table, and a bar chart (overall + per super-category). The best row is
  **printed as a suggestion**. Copying it into `BASE` is a manual, reviewed change.

Resolution cost (patch 14; attention grows with tokens²):

| image_size | grid | tokens | tokens vs 644 | attention vs 644 |
|---|---|---|---|---|
| 644 | 46×46 | 2116 | 1.00× | 1.0× |
| 756 | 54×54 | 2916 | 1.38× | 1.9× |
| 840 | 60×60 | 3600 | 1.70× | 2.9× |
| 952 | 68×68 | 4624 | 2.19× | 4.8× |

Evidence: fine-tuned DINOv2 on SPair-71k reaches 81.7 at 840 px vs 76.6 at 448 and 58.2 at 224 [1]. GeoAware-SC
also uses 840 px (60×60) [2]. A higher resolution mostly fixes near misses, not swaps or far misses.

---

## 5. Phase 3: Training on the Colab T4

### 5.1 How to start it
`colab/run.ipynb` with `MODE = "train"`. Settings come from `BASE` on the pushed branch (values in §6).
After training: `evaluate --run <run_id>`, then `submit --run <run_id>`.

**Fix for ViT-L:** `head_dim` is 768. With ViT-L (1024-d), the `linear` head would **not** start as the
identity (random 1024→768 projection), so training would start below the baseline. `head_dim=None` should
mean "same as the backbone", and become the default.

### 5.2 T4 facts that shape the choices

| T4 property | Consequence |
|---|---|
| 16 GB memory, ~320 GB/s | memory is fine for ViT-B; ViT-L at ≥ 840 px needs a small micro-batch |
| fp16 tensor cores, **no bf16** | AMP in fp16 + `GradScaler` (already in `train.py`); keep softmax/loss in fp32 (already `.float()`) |
| sm75: no FlashAttention-2 in PyTorch | use SDPA's memory-efficient kernel (`attn_implementation="sdpa"` in transformers). Check at start-up that the math kernel is not used, because it stores the full N×N attention matrix (ViT-L @ 952: ~0.7 GB per block per image) |
| Colab free tier disconnects | `checkpoint_every_steps` ≈ every 10 minutes of training; resume with `RESUME` |
| 2 CPU cores | `num_workers=2`; images are decoded on the CPU, so keep augmentation light |

Rough time per epoch on ~1,800 training queries (order of magnitude; the first run gives the real number):
- **ViT-B @ 644, 2 blocks:** ≈ 0.7 TFLOP per image (full forward + backward through 2 blocks), 2 images per
  query, ~20 TFLOPS sustained in fp16. That is **≈ 2–4 min/epoch**, plus ~1 min of validation.
- **ViT-L @ 644:** ≈ 3.5× ViT-B, so **≈ 8–12 min/epoch**.
- **ViT-L @ 952:** ≈ 9× ViT-B, so **≈ 25–35 min/epoch**. Only worth it if the sweep shows a clear gain.

### 5.3 Batch size: automatic, and why
**What uses memory.** One sample = 2 images.
- Weights: ViT-B fp16 ≈ 0.2 GB, ViT-L ≈ 0.6 GB.
- AdamW state for the trainable part: 16 bytes per parameter. 2 ViT-B blocks ≈ 14M parameters ≈ 0.23 GB;
  2 ViT-L blocks ≈ 25M ≈ 0.4 GB.
- Activations of the **trainable blocks only**: frozen blocks keep nothing, because nothing before them needs
  gradients. That is about `20 × tokens × width × 2 bytes` per block per image: ViT-B @ 644 ≈ 65 MB, so
  2 blocks × 2 images ≈ 0.26 GB per sample, and **a micro-batch of 16 fits in ~5 GB**.

So on the T4 the batch is limited by **time**, not memory, for ViT-B. For ViT-L at high resolution it is
limited by memory.

**Proposal:**
1. `batch_size="auto"`: at start-up, run a dummy forward+backward step with the real settings at
   micro-batch 1, 2, 4, 8, 16 until the GPU runs out of memory. Keep the largest that worked × 0.85.
2. `effective_batch=16` is what the optimiser sees. `grad_accum_steps = ceil(effective_batch / micro_batch)`.
   The LR and schedule depend only on `effective_batch`, so a T4 run and an A100 run are comparable.
3. Log `micro_batch`, `grad_accum_steps`, `effective_batch` and peak memory to W&B and `config.json`.

**Why separate them:** the micro-batch is a speed choice (largest that fits). The effective batch is a learning
choice (gradient noise vs. number of updates). With LayerNorm (no BatchNorm), the micro-batch does not change the result.

**Why 16 and not larger:** with ~1,800 queries, 16 gives ~113 updates per epoch and ~800 in 7 epochs. With 64
there would be only ~200 updates in total, too few for warmup + decay. GeoAware-SC even trains with batch 1 [2].

### 5.4 Batch size that changes over the epochs
It is possible: change `grad_accum_steps` per epoch (`effective_batch_schedule`, e.g. `{0: 8, 3: 16}`), with the
LR schedule driven by **samples seen** instead of steps. A growing batch acts like LR decay (Smith et al., 2017 [6]).
**Recommendation: don't use it here.** The cosine decay (§6.4) already lowers the gradient noise at the end,
and ~800 updates are too few to gain from a second mechanism that interacts with the LR. Implement it only as an
option, and only after a constant-batch run.

### 5.5 Data augmentation (PIL images, train only, source and target separately)

| Augmentation | Proposal | Notes |
|---|---|---|
| pair flip | p = 0.5 (exists) | both images flipped, so left/right labels stay consistent; GeoAware-SC's main augmentation [2] |
| rotation | ±15°, p = 0.5 | points rotated with the same matrix; edge rule §3.3 |
| colour jitter | brightness/contrast/saturation 0.3, hue 0.05, p = 0.8 | points unchanged |
| grayscale | p = 0.1 | |
| random resized crop | scale 0.7–1.0, p = 0.5 | must keep all targets (§3.3) |
| keypoint jitter | ±0.5 cell on the **target label** only | regulariser, as in GeoAware-SC (Gaussian noise on ground-truth keypoints [2]) |

- One function `augment_sample(sample, cfg, rng)`. It is seeded, so `--resume` gives the same batches.
- The on-disk feature cache cannot be used with augmentation. Turn it off automatically and say so in the log.
- Validation and test never use augmentation.

### 5.6 W&B charts

| Group | Charts (per step `train/*`, per epoch `val/*`) |
|---|---|
| Loss | `train/loss`, `train/loss_ema`, **`train/loss/<super-category>`** ×4, `val/loss`, `val/loss/<super-category>` |
| Learning rate | `lr/head`, `lr/backbone`, `lr/block_<i>` (with layer decay) |
| Batch | `batch/micro`, `batch/effective`, `batch/grad_accum_steps`, `samples_seen` |
| Optimisation | `grad_norm` (before clipping), `grad_clip_fraction`, `amp_scale`, non-finite count, `train/softmax_entropy` (is the matching distribution sharpening?) |
| Validation | `val/pck@0.05/0.10/0.15`, **`val/pck@0.10/<super-category>`** ×4, `val/near_miss_pct`, `val/swap_pct`, `val/far_miss_pct`, `val/best_pck` |
| Tables | per-category PCK (worst 15, with counts) each epoch; sweep table (evaluate) |
| Images | 8 worst validation predictions per epoch |
| System | peak GPU memory, samples/s, seconds per epoch |

The per-super-category loss is cheap: the loss is already computed per sample, so group by super-category before averaging.

---

## 6. Hyperparameters in `src/config.py`: the training decisions

### 6.1 Why the number of training queries matters
Many choices below depend on **N = number of training queries**:

| Choice | How it depends on N |
|---|---|
| updates per epoch | N / effective_batch |
| warmup length and total schedule | measured in updates, so fixed by N × epochs / batch |
| number of epochs | small N → more epochs needed for enough updates, but a higher risk of overfitting |
| blocks to unfreeze, LR | small N → fewer trainable parameters and a lower LR (the model can memorise) |
| batch-size ramp | only useful with many updates (§5.4) |
| T4 time budget | time per epoch is proportional to N |
| how noisy validation is | fixed at 453 queries, but it limits how many settings can be compared (§6.10) |

**Measured on the real data:** validation = 78 pairs = **453 queries**, training split = **2,042 queries**
(392 − 78 = 314 pairs). That is ~25× fewer than SPair-71k's ~53k training pairs, so the recommendations below
are **more conservative** than the papers. At effective batch 16: 127 updates per epoch, ~890 in 7 epochs.

### 6.2 Priority order
Tune in this order. The first items matter most:
1. **What is trained** (blocks) and **backbone LR**: decide whether fine-tuning helps or destroys the features.
2. **image_size / model**: from the evaluation sweep (§4.3), *before* training.
3. **Loss temperature and label width**: shape of the matching distribution.
4. **Epochs and schedule.**
5. **Augmentation.**
6. Batch, weight decay, gradient clipping: second order, keep the defaults.

### 6.3 Learning rate

| Key | Now | Proposal | Range to try | Why |
|---|---|---|---|---|
| `lr_backbone` | 1e-5 | **1e-5** | 5e-6, 1e-5, 2e-5 | see below |
| `lr_head` | 1e-4 | **1e-4** (10× backbone) | 1e-4 … 1e-3 | the head is small and starts as the identity; it can move faster |
| `layer_decay` (new) | — | **0.75** (ViT-B), 0.8 (ViT-L) | 0.65–0.85 | only matters with ≥ 3 trained blocks |

**Why ~1e-5 for the backbone:**
- MAE fine-tunes ViT-B with base lr 5e-4 × batch/256 and layer decay 0.65–0.75 [3]. At our effective batch 16
  that is ≈ 3e-5 for the top block. Lower blocks get less.
- Our data is ~30× smaller than those benchmarks and the features are already good (40.4 zero-shot). A step 2–3×
  lower (**1e-5**) protects them. The Lang pipeline used the same value.
- 5e-6 is the "safe" option if val PCK drops in the first epoch. 2e-5 is the "fast" option if it barely moves.
- For comparison, GeoAware-SC trains **only a small head on frozen features** with a peak lr of 1.25e-3 [2]. That
  sets the scale for the head, not for the backbone.

**Layer-wise LR decay (LLRD):** block `i` from the top gets `lr_backbone × layer_decay^i`. BEiT/MAE use 0.65 for
ViT-B and 0.75–0.8 for ViT-L [3][4]. With 2 blocks the effect is small (block 11 gets 0.75× block 12). With 3–4
blocks it keeps the lower blocks close to their pretrained values. Implement it as one parameter group per block.

**No weight decay** on LayerNorm weights and biases (standard practice). The current code decays every parameter.

### 6.4 How the learning rate decreases over training

| Option | Verdict |
|---|---|
| **Linear warmup + cosine decay, per step** | **Recommended.** Smooth, no extra settings, works well for short fine-tuning |
| OneCycleLR (now) | Same shape, but PyTorch's defaults are not what we want: the start is peak/25 and the end peak/(25 × 10⁴) ≈ 0, and `cycle_momentum=True` also changes AdamW's β1 between 0.85 and 0.95. Keep it only with `cycle_momentum=False`, `div_factor=10`, `final_div_factor=10` |
| Step decay per epoch (e.g. ×0.1 at epoch 5) | Coarse; adds a choice (when) with few epochs to place it in |
| ReduceLROnPlateau on val PCK | Bad here: val is noisy (±2.3 points) and there are only 6–8 evaluations |
| Constant LR | Simple baseline, but no annealing at the end |

**Proposal:**
- `lr_schedule="cosine"`, `warmup_pct=0.10` (with a minimum of 30 updates), `min_lr_ratio=0.01`.
- Update the LR **every optimiser step**. "Decreasing along the epochs" then happens smoothly inside each epoch, not in jumps.
- Formula, with `t` = update number and `T` = total updates:
  `lr(t) = peak × t / T_warm` for `t < T_warm`, then
  `lr(t) = peak × [min_lr_ratio + (1 − min_lr_ratio) × 0.5 × (1 + cos(π (t − T_warm) / (T − T_warm)))]`.
- **Why warmup:** the first updates use AdamW's still-poor moment estimates and a random-ish gradient direction.
  At the full LR they can damage the pretrained blocks in a few steps. 10% of ~800 updates = 80 updates.
- **Why cosine to 1%, not to 0:** the last epochs still learn a little. Since `best.pt` is kept anyway, the exact end value matters less.
- The schedule is planned for all `epochs`. With early stopping, training stops earlier but the schedule does not
  shrink, so **choose `epochs` to fit the budget** (§6.6).

### 6.5 How many blocks to fine-tune

| Key | Now | Proposal | Try |
|---|---|---|---|
| `unfreeze_blocks` | 2 | **2** (ViT-B) / **4** (ViT-L) | ViT-B: 1, 2, 3 · ViT-L: 2, 4, 6 |
| `feature_layer` | −1 | −1 | must be **after** the last frozen block; otherwise the trained blocks are not used |

**Evidence:** fine-tuning DINOv2 at 840 px on SPair-71k gave 81.7 (last block only), **85.1 (last 2 blocks)**,
84.5 (3), 84.0 (4) and 82.2 (5) PCK@0.10 [1]. More blocks were *worse*, even with ~53k training pairs.

**Why the last blocks:** the last blocks of a ViT hold the most task-specific (semantic) features. The lower blocks
hold generic local features that the matching also needs. Changing few parameters (2 ViT-B blocks ≈ 14M of 86M)
limits overfitting on ~1,800 queries.

**Tips:**
- ViT-L has 24 blocks: the same *fraction* (2/12 ≈ 17%) is 4 blocks.
- Always train the final LayerNorm with the blocks (already done in `unfreeze_last`).
- If 1 block ≥ 2 blocks on val, the data is too small for more: stop there and consider LoRA (rank 8 beat a
  linear head on SPair-71k in DistillDIFT [5]) as Stage 3.
- Watch `val/far_miss_pct` and the per-category table: if they get worse for categories that were good zero-shot,
  the features are being damaged. Lower the LR or the number of blocks.

### 6.6 Epochs and early stopping

| Key | Now | Proposal | Why |
|---|---|---|---|
| `epochs` | 2 | **7** | ≈ 800 updates at batch 16 with N ≈ 1,800. GeoAware-SC needs ~2 epochs, but on ~30× more data [2] |
| `val_every_epoch` | True | True | `best.pt` = best val epoch, which is the early stopping |
| `early_stop_patience` (new) | — | **3** epochs | stop when val PCK has not improved for 3 epochs; saves T4 time |
| `checkpoint_every_steps` | 0 | **≈ 10 minutes of training** (e.g. 150 for ViT-B @ 644) | Colab disconnects |

Overfitting signal in W&B: `train/loss` keeps falling while `val/pck@0.10` is flat or falls for 2 epochs.

### 6.7 Loss: temperature and label width

| Key | Now | Proposal | Try | Why |
|---|---|---|---|---|
| `loss_temperature` | 0.03 | 0.03 | 0.02, 0.03, 0.05 | Features are L2-normalised, so similarities are in [−1, 1]. Without a low temperature the softmax is too flat and the gradients are weak (SimSC [7]). 0.03 is the Lang value |
| `label_sigma` (cells) | 1.0 | 1.0 | 0.5, 1.0, 2.0 | Width of the Gaussian around each accepted target. Smaller = sharper but noisier |
| `learn_temperature` (new, optional) | — | off | on | SimSC learns it with the backbone [7]; useful if the temperature sweep is flat |

The inference settings `soft_temperature` / `soft_window` are separate and come from the evaluation sweep. **Re-run
the matching sweep after fine-tuning**, because the trained features are sharper.

### 6.8 Regularisation and optimiser

| Key | Now | Proposal | Why |
|---|---|---|---|
| optimiser | AdamW | AdamW, β = (0.9, 0.999) | standard; no momentum cycling (§6.4) |
| `weight_decay` | 0.05 | **0.05**, excluding LayerNorm/bias | at lr 1e-5 its pull is tiny; GeoAware-SC's head uses 0.001 [2] |
| `grad_clip` | 1.0 | 1.0 | protects against fp16 spikes; log how often it triggers |
| `amp` | True | True (fp16 on the T4) | |
| `flip_augment_prob` | 0.5 | 0.5 | |
| augmentation | flip only | §5.5 | |
| `seed` | 0 | 0, plus seeds 1 and 2 for the final choice | §6.10 |

### 6.9 Summary of the `BASE` changes

```python
# Training (Colab T4)
"head": "linear", "head_dim": None,            # None = same as the backbone (identity start, also ViT-L)
"train_mode": "finetune", "unfreeze_blocks": 2,
"lr_backbone": 1e-5, "lr_head": 1e-4, "layer_decay": 0.75,
"lr_schedule": "cosine", "warmup_pct": 0.10, "min_lr_ratio": 0.01,
"batch_size": "auto", "effective_batch": 16, "effective_batch_schedule": None,
"epochs": 7, "early_stop_patience": 3, "checkpoint_every_steps": 150,
"loss_temperature": 0.03, "label_sigma": 1.0, "weight_decay": 0.05, "grad_clip": 1.0,
"aug_rotation_deg": 15, "aug_color_jitter": 0.3, "aug_crop_scale": [0.7, 1.0], "aug_edge_margin": 14,
# Evaluation sweeps (report only)
"sweep_flip_test": [False, True], "sweep_soft_window": [3, 5, 7, 9, 11], "sweep_soft_temperature": [0.02, 0.04],
"sweep_image_size": [], "sweep_model_id": [],
```

### 6.10 Tuning protocol for a small validation set
- **Noise:** with 453 queries and PCK ≈ 50%, the standard error is ≈ 2.3 points (95% interval ± 4.6). Per
  super-category it is larger: Furniture may have only ~40 queries (± 15 points). Per category (~4–5 queries each)
  the numbers are **anecdotes**, not measurements.
- **Rule:** accept a change if it gains ≥ 3 points overall, or if it gains with 2 seeds out of 2.
- **Don't overfit the validation set:** every setting we compare is chosen on the same 453 queries. Keep the number
  of compared runs small (§6.11), and use the Kaggle leaderboard only to confirm the final choice.
- **Order:** one parameter at a time, the others at the proposal value, from the top of the priority list (§6.2).

### 6.11 Run plan on the T4 (≈ 6 GPU-hours)

| Run | Change from the proposal | Purpose | Est. time |
|---|---|---|---|
| E0 | evaluate + matching sweep + feature sweep (ViT-B 644/756/840, ViT-L 644) | pick model/resolution | ~1 h |
| T1 | proposal (ViT-B, best size, 2 blocks, 1e-5) | does fine-tuning help? | ~30 min |
| T2, T3 | `lr_backbone` 5e-6 / 2e-5 | LR | ~1 h |
| T4, T5 | `unfreeze_blocks` 1 / 3 | blocks | ~1 h |
| T6 | `loss_temperature` 0.02 or 0.05 (direction from T1's entropy curve) | loss | ~30 min |
| T7 | full augmentation on vs flip only | augmentation | ~30 min |
| T8 | best setting, seeds 1 and 2 | confirm | ~1 h |
| (T9) | ViT-L, 4 blocks, best size ≤ 840 | only if E0 shows ViT-L ≥ +3 zero-shot | ~1.5 h |

---

## 7. Phase 4: Submission (unchanged)
`python run.py --mode submit --run <run_id>`: same files, same checks (`check_submission`), same format. Sweep
settings are ignored, and training-only code is not imported.

## 8. Colab: Miniconda or pip?
**Recommendation: keep pip** (current `scripts/colab_setup.py`, which installs the `pip:` list of `environment.yml`).

| | Miniconda on Colab (`condacolab`) | pip from `environment.yml` (now) |
|---|---|---|
| Setup time per session | +3–6 min (install, **kernel restart**, solve) | ~1 min |
| torch + CUDA | download a ~2.5 GB CUDA torch every session | Colab's preinstalled CUDA torch |
| Kernel restart | yes: variables and the Drive mount must be redone | no |
| Same versions as local | exact | same pins except torch/torchvision |

Colab keeps nothing installed between sessions, so conda would rebuild everything every time, for an exact torch
match the code does not need. Cheap improvement instead: `colab_setup.py` prints a **version check** (installed vs
`environment.yml`) for torch, transformers, numpy and pandas.

## 9. Files that change

| File | Change |
|---|---|
| `src/models/matching.py` | edge-safe window (§3.1) |
| `data/transforms.py` | `augment_sample`: rotation, colour, crop, edge rule |
| `data/datasets.py`, `data/supercategories.json` | `supercategory` in every sample |
| `src/train.py` | auto micro-batch, effective batch, cosine schedule, layer decay, no decay on LN/bias, early stopping, per-sample loss, W&B |
| `src/utils/batch_size.py` (new) | out-of-memory probe |
| `src/models/backbone.py` | `attn_implementation="sdpa"` + kernel check; `head_dim=None` support in `src/models/__init__.py` |
| `src/evaluate.py`, `src/diagnostics.py`, `src/metrics.py` | breakdowns, error types, confidence intervals, sweeps, W&B |
| `src/config.py` | keys of §6.9 |
| `scripts/colab_setup.py` | version check |
| `run.py`, `src/predict.py` | **no change** |

## 10. Verification
1. Edge fix: val PCK ≥ 40.4. Show the edge-bucket table before and after.
2. Augmentation: draw 16 augmented pairs with their points and check them by eye; no point on padding.
3. Batch probe: report the micro-batch found for ViT-B@644 and ViT-L@644 on the T4. One training step at that size must not run out of memory.
4. Schedule: the `lr/*` chart shows warmup then cosine down to 1%, with no momentum cycling.
5. Resume: stop a run in the middle of an epoch, resume it, and compare the W&B loss curve with an uninterrupted run.
6. Submission: `submit` of the baseline gives a byte-identical `submission.csv`, and `check_submission` passes.

## 11. Open questions
1. ~~Number of training queries~~: 2,042 (status section).
2. ~~Categories~~: none in the data; super-categories skipped (user decision).
3. Is W&B the shared team project, and should evaluate runs be logged there too? (`wandb_evaluate=True` now.)

## Sources
1. [Semantic Correspondence: Unified Benchmarking and a Strong Baseline (2025)](https://arxiv.org/abs/2505.18060): resolution and number-of-blocks ablations for fine-tuned DINOv2 (Tables VI–VII).
2. [Telling Left from Right: Geometry-Aware Semantic Correspondence (CVPR 2024)](https://arxiv.org/abs/2311.17034): AdamW, one-cycle at 1.25e-3 for a head on frozen features, batch 1, 840 px DINOv2, flip augmentation, keypoint noise, window soft-argmax. Code: [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC).
3. [Masked Autoencoders Are Scalable Vision Learners (MAE)](https://arxiv.org/abs/2111.06377): fine-tuning recipe, layer-wise decay.
4. [BEiT v2](https://arxiv.org/abs/2208.06366) and [mc-BEiT](https://arxiv.org/abs/2203.15371): layer decay 0.65 (ViT-B), 0.75–0.8 (ViT-L).
5. [Distillation of Diffusion Features for Semantic Correspondence (DistillDIFT)](https://arxiv.org/abs/2412.03512): LoRA rank 8 vs linear head.
6. Smith et al., [Don't Decay the Learning Rate, Increase the Batch Size (2017)](https://arxiv.org/abs/1711.00489).
7. [SimSC: A Simple Framework for Semantic Correspondence with Temperature Learning](https://arxiv.org/abs/2305.02385).
8. [PyTorch `scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): FlashAttention-2 / memory-efficient / math kernels.
