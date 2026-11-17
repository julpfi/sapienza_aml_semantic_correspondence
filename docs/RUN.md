# RUN.md: how to run the project

How to set up the project, run every step and submit, locally, in Colab or on Kaggle; how the "run" files fit
together; the fine-tuning hyperparameters; and how to read the W&B charts.
The analysis of the first fine-tuning run is in [REVIEW_finetuning.md](REVIEW_finetuning.md).

## 1. What the pipeline does

For each query (source image, target image, point on the source), it predicts the matching
point on the target. The score is PCK@0.10 (see `.claude/CLAUDE.md`).

```text
image ──► square resize 644 px ──► DINOv2-B/14 (facebook/dinov2-base) ──► 46×46 patch features
      ──► (D-008: concatenated with 0.5 × frozen Stable Diffusion 2.1 features, `fusion` "concat")
      ──► cosine similarity of the source point vs every target cell
      ──► 2× upsampling + 11×11 soft-argmax (T = 0.02) ──► (x, y) in original target pixels
```

There is one configuration: `BASE` in `src/config.py`. It is set up for fine-tuning on a Colab T4
(`train_mode="finetune"`, last two DINOv2 blocks, no head, as in Lang). Without a trained `best.pt`,
`evaluate`/`submit` give the frozen DINOv2 baseline (val PCK@0.10 = 40.4 with
the old 5×5 window, 41.1 with the 11×11 window, checked 2026-11-05). A trained model is used with `--run <run_id>`.
Hyperparameters: section 8.

| What | Where to run |
|---|---|
| `evaluate` / `submit` (baseline or `--run`) | anywhere; CPU works but takes ~1 h for validation |
| `train` | Colab T4 (`--require-gpu` in the notebook) |

Everything goes through one command: `python run.py --mode train|evaluate|submit`.

## 2. The "run" files: why there are several

> **A run** is one execution of one mode (`train`, `evaluate`, `submit`) with one config. It gets an id
> (`20261103-165509_evaluate`) and its own folder, `outputs/<run_id>/`, which holds the settings that
> produced it and everything it wrote.

The team works on a laptop, Colab (GPU, training) and Kaggle (offline, submission notebook). The project needs
one way to run the code on all three, and one record per run, so every score can be traced back to its config,
code version and checkpoints, even after a Colab disconnect. The files below are **not duplicates**:

| File | What it is for |
|---|---|
| `run.py` | **The only entry point.** Parses `--mode/--run/--resume`, checks files and device (`preflight`), creates the run folder, builds the model, calls train / evaluate / submit |
| `src/utils/run.py` | **Run folder logic:** `prepare_run_dir` (creates or reopens `outputs/<run_id>/`, writes `config.json` once), `load_run_config` (reloads a run's settings; keys added to `BASE` later get their defaults), `git_commit` (`-dirty` with uncommitted changes), `seed_everything` |
| `docs/RUN.md` | This guide |
| `colab/run.ipynb` | **Thin Colab runner.** One cell sets `BRANCH/MODE/RUN/RESUME`, then `scripts/colab_setup.py` and `python run.py ...`. No logic |
| `notebooks/kaggle_run.ipynb` | **Thin Kaggle runner (internet off).** Sets the weights folder and offline mode, then `python run.py --mode submit` |
| `scripts/colab_setup.py` | Prepares a Colab session **before** `run.py`: pull, install, secrets, Drive, data, weights |
| `scripts/kaggle_io.py` | Downloads the competition data; uploads a run's `submission.csv` |
| `scripts/download_weights.py` | Saves the DINOv2 weights locally for offline runs |
| `src/config.py` | `BASE` and the runtime paths; `output_dir` decides where run folders go |

```text
          laptop                         Colab                             Kaggle (offline)
            │                    colab/run.ipynb                     notebooks/kaggle_run.ipynb
            │                    └─ scripts/colab_setup.py           └─ sets SC_WEIGHTS_DIR, HF_HUB_OFFLINE
            ▼                              ▼                                    ▼
  ┌──────────────────────────── python run.py --mode ... ───────────────────────────────┐
  │ parse_args → load_config (BASE, or the saved config.json for --run/--resume)          │
  │ preflight (files, GPU) → seed_everything → prepare_run_dir → build_model              │
  └─────────────┬───────────────┬────────────────┬─────────────────────────────────────┘
                ▼               ▼                ▼
          src/train.py    src/evaluate.py   src/predict.py
                └───────────────┴────────┬───────┘
                                         ▼
                              outputs/<run_id>/   (Drive on Colab, /kaggle/working on Kaggle)
```

The notebooks and scripts only **prepare the machine**; `run.py` and `src/` do **the work**, with the same
code everywhere. `src/config.py` detects the runtime and resolves `data_dir`, `output_dir`, `weights_dir`,
`features_dir`.

## 3. One-time setup (local)

```bash
# 1. Environment (Miniconda). environment.yml is the only dependency file.
conda env create -f environment.yml
conda activate advance
#    CPU-only Linux machine? Swap in the smaller CPU build of torch:
#    pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu

# 2. Secrets: copy the template and fill in your keys (.env is never committed).
cp .env.example .env          # HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY, WANDB_API_KEY

# 3. Competition data -> data/raw/ (accept the rules on kaggle.com first).
python scripts/kaggle_io.py download

# 4. Optional: keep the DINOv2 weights in weights/dinov2-base (offline use, Kaggle upload).
python scripts/download_weights.py
```

If `conda` is not found, run `~/miniconda3/bin/conda init bash` once and open a new terminal.
After a change to `environment.yml`: `conda env update -n advance -f environment.yml --prune`.
Check the setup without running anything: `python run.py --mode submit --dry-run`.

## 4. The three modes and the run folder

Every command creates (or reopens) one run folder `outputs/<run_id>/`, with `<run_id>` =
`<UTC time>_<mode>`. `config.json` (every setting plus the git commit) is always the first file written.

```text
outputs/
  20261104-120000_evaluate/
    config.json        settings of this run
    summary.json       PCK@0.05/0.10/0.15 (+95% interval), score, error mix, PCK and loss per group
    per_query.csv      one row per validation query (prediction, dist = error / L, groups, error type)
    sweep.csv          flip_test x soft_window x soft_temperature (x image_size x model): a report only
    wandb/             only for offline W&B runs (to sync later); online runs keep these files off Drive
  20261104-130000_train/
    config.json
    history.json       validation PCK, loss and error mix after every epoch
    checkpoints/       best.pt (best validation PCK); last.pt only while training (deleted at the end)
  20261104-140000_submit/
    config.json
    predictions.csv    id,x,y
    submission.csv     id,prediction (the Kaggle file)
```

| Command | What it does |
|---|---|
| `python run.py --mode evaluate` | Validation PCK, groups and error types (printed), sweep, W&B (no image: the worst-prediction plot is drawn by `train` only) |
| `python run.py --mode train` | Train with `BASE` (validation every epoch, best kept) |
| `python run.py --mode train --resume <run_id>` | **Reopen** the folder and continue from `last.pt` (weights, optimiser, scheduler, epoch and batch position, random state) |
| `python run.py --mode evaluate --run <run_id>` | **Reopen** a trained run, load `best.pt`, write the evaluation files into it |
| `python run.py --mode submit [--run <run_id>]` | Test predictions + Kaggle file, checked against `sample_submission.csv` |

Other flags: `--require-gpu` stops if there is no CUDA GPU; `--dry-run` only prints the settings and checks
the files.

Rules that keep runs comparable:

- `config.json` is written **once**; reopening a run never overwrites it.
- `--run` and `--resume` use the run's saved `config.json`, **not** the current `BASE`. To score an old run
  with new inference settings, edit that run's `config.json` (keep a copy). A change of settings for
  training needs a new run.
- When a run is reopened on a different commit, `run.py` prints the old and the new commit.
- Seeds and `cudnn.deterministic` are set before every run.

## 5. Typical workflow

1. **Validate:** `python run.py --mode evaluate`. The split is `splits/val_pair_ids.json` (453 queries).
2. **Look at the failures:** `per_query.csv` (`dist` = error / L); for a trained run, `val/worst_predictions` on W&B.
3. **Submit:**
   ```bash
   python run.py --mode submit                    # or --run <run_id> for a trained model
   python scripts/kaggle_io.py submit --file outputs/<run_id>/submission.csv --message "baseline: val 40.4"
   ```
4. **Train** on Colab (section 6) with `MODE = "train"`. Evaluate and submit with `--run <run_id>`.
5. **Sweep:** `evaluate` writes `sweep.csv`. Copy a setting into `BASE` only if its gain is larger than its
   95% interval (`ci95`). The submission never reads `sweep.csv`. For resolutions / ViT-L, fill
   `sweep_image_size` / `sweep_model_id` in `BASE` and run evaluate on the T4.
6. **Check the augmentation** after changing an `aug_*` setting: `python scripts/check_augmentation.py`
   (draws augmented pairs and checks that no point is moved next to the border).

Before every submission (CLAUDE.md): `submit` checks that every id of `sample_submission.csv` appears exactly
once, in order, with finite values, and clips points to the image. Still look at `val/worst_predictions` after
a change to resizing or coordinates.

## 6. Colab (training)

Open `colab/run.ipynb` in Colab (File > Open notebook > GitHub, this repo).

1. Runtime > Change runtime type > **GPU**.
2. Colab Secrets (key icon), with notebook access: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `WANDB_API_KEY`, `HF_TOKEN`.
3. In the settings cell, set `BRANCH`, `MODE` (and `RUN` or `RESUME`), then run all cells. The run uses `BASE`
   from the pushed branch.

- `%run scripts/colab_setup.py` pulls the branch, pip-installs `environment.yml` (keeping Colab's CUDA torch),
  loads the secrets, mounts Drive, downloads the data to `/content/data/raw` and saves the weights to Drive.
- Runs are saved to `MyDrive/aml_semantic_correspondence/outputs/<run_id>`, so they survive a disconnect.
- **After a disconnect:** set `RESUME = "<run_id>"` and run all cells again. `last.pt` is saved at the end of
  every epoch (`checkpoint_every_steps = 0`; set a number of updates to also save inside an epoch) and is
  deleted when the training ends, to keep Drive small. Drive keeps overwritten files in its bin for 30 days,
  so empty the bin if the space runs out.
- The last cell can upload a `submission.csv` to Kaggle.

Push your code before running it: Colab pulls from GitHub, and edits made in Colab are lost.

## 7. Kaggle (offline submission notebook)

1. Locally: `python scripts/download_weights.py`, then upload `weights/` as a Kaggle dataset (with `dinov2-base/`).
2. Upload the repo as a Kaggle dataset, without `data/raw`, `weights/` or `outputs/`.
3. Create a notebook from `notebooks/kaggle_run.ipynb`, attach the competition data, the repo and the weights
   datasets, and turn the internet off.
4. Set `REPO_DIR` and `WEIGHTS_DIR` in the first code cell and run all. `--mode submit` writes
   `/kaggle/working/submission.csv`.

## 8. Fine-tuning hyperparameters

Current `BASE` (since D-008; the v2 value is given where it changed, v2 = `BASE` + `V2_TRAINING`). Run 1 is `20261104-155818_train` (val PCK@0.10 62.9); the numbers in the last column are
sections of [REVIEW_finetuning.md](REVIEW_finetuning.md).

| Setting | Now | Notes |
|---|---|---|
| **What is trained** | | |
| `train_mode` / `head` | finetune / **none** (was linear) | as in Lang: blocks 10–11 already adapt the features; a 768×768 head at lr 1e-3 changed every patch at once and could flatten the maps. Update 0 = frozen DINOv2 |
| `unfreeze_blocks` | 2 (blocks 10 and 11 + final LayerNorm) | 2 was the best setting for ViT-B on SPair-71k (1.1) |
| **Optimiser** | AdamW + OneCycleLR | |
| `lr_backbone` (peak) / `layer_decay` | **1.5e-5** (was 1e-5; block 11 and final LayerNorm) / 0.75 | block i from the top gets lr_backbone × 0.75^i: block 10 at 1.125e-5 |
| `weight_decay` | 0.05, on every trained parameter | plain AdamW |
| **Schedule** | | |
| `epochs` / `early_stop_patience` | **8** (v2: 5) / 3 | pair mode: ≈ 78 updates per epoch, so 8 epochs ≈ the v2 total |
| `effective_batch` / `pair_effective_batch` | **16** (was 8) per-query mode / **4 pairs** (≈ 25 queries) in pair mode, micro-batch auto | ≈ 127 updates per epoch, ≈ 638 in 5 epochs: less noisy gradients than at 8. Optional ramp: `effective_batch_schedule` |
| `warmup_epochs` | **1.5** (v2: 1.0; replaces `warmup_pct` 0.10) | OneCycleLR warmup in epochs: lr rises from peak / 25 to the peak over the whole of epoch 0 (≈ 128 updates), then cosine to peak / 25e4. Independent of `epochs`. The peak lr is the last warmup update. Slower warmup against the entropy rise of the first epoch (1.8) |
| `grad_clip` | **10.0** (was 5.0, then 7.0) | max total gradient norm. The norm scales with 1 / T: ≈ 11 at T = 0.03 (run 1), so expect ≈ 7 at 0.05. At 5 the clip is a spike guard that still lets the gradient size vary; at 1 every update was clipped (1.9) |
| precision | fp32 (no AMP) | as in the Lang pipeline; slower and more memory than fp16, so the auto micro-batch is smaller |
| **Loss** | | |
| `loss` | **`marginal`** (v2: `soft_ce`) | `soft_ce`: cross-entropy with Gaussian bumps on all accepted targets (entropy floor ≈ 3.3–3.5). `marginal` (Lang): −log of the probability on the cells within `marginal_radius` (1 cell) of any target; lets the model commit to one target with a sharp peak. raw losses are on different scales, so only `loss_over_chance` is charted |
| `loss_temperature` | **0.04** (was 0.05) | both losses; higher = flatter softmax and gradients 2.5× smaller than at 0.02, for a calmer first epoch. Loss and entropy charts are not comparable across temperatures |
| `loss_temperature_start` | **0.07** (on) | the loss temperature starts there and falls linearly to `loss_temperature` over `loss_temperature_epochs` = **3.0** (v2: 2.0) (`train/loss_temperature`). Validation always uses `loss_temperature` |
| `train_on_val` | False | final submission only: train on the validation pairs too (2,495 queries instead of 2,042). No validation, no early stop, `best.pt` = last epoch: set `epochs` from a validated run first |
| `label_sigma` | 1.0 cell | `soft_ce` only |
| `label_jitter` | **0.25** cell (v2: 0.5) | noise on the target labels; small objects got worse in run 1 |
| **Augmentation** (every training image) | | |
| `flip_augment_prob` | 0.5 (both images) | keeps left/right consistent |
| `aug_crop_scale` / prob | **[0.8, 1.0]** (v2: [0.7, 1.0]) / 0.5 | |
| `aug_rotation_deg` / prob | **10°** (v2: 15°) / 0.5 | |
| `aug_color_jitter` / `aug_hue_jitter` / prob | **0.2 / 0.02 / 0.5** (v2: 0.3 / 0.05 / 0.8) | |
| `aug_grayscale_prob` / `aug_blur_prob` | **0.05 / 0.05** (v2: 0.1 / 0.1) | |
| **Inference** (validation each epoch and submission) | | |
| `soft_window` / `soft_temperature` | **11 / 0.02** (v2: 9 / 0.04; +3.1 points on run 1's model) | the sweep in `evaluate` scores other values; re-run it after the first marginal run |
| `flip_test` | False | |

## 8b. v3 additions (branch sd)

Every new key is in `BASE`. **Since D-008 the defaults below are the v2 values, not `BASE`:** `BASE` holds
the recommended stage-1 setting plus `fusion` "concat" ([decisions.md](decisions.md) D-008); v2 = `BASE` updated with
`V2_TRAINING` (`src/config.py`) for training; the matching keys (`soft_window` 11, `soft_temperature` 0.02) stay as
in `BASE` on purpose, so every run is compared at the same matching settings. The "`BASE` now" column gives the
current value. The plan is
[PLAN.md](PLAN.md); the choices are in [decisions.md](decisions.md). The run order for Colab is in
[experiments-sd.md](experiments-sd.md).

| Setting | v2 value | What it does | When to turn it on |
|---|---|---|---|
| `pair_batch` | False (`BASE` now: True) | one training item = one image pair with all its queries: one backbone pass supervises about 6 queries | with `hard_neg_weight`; also for speed (about 78 updates per epoch at `pair_effective_batch` 4 instead of about 127) |
| `pair_effective_batch` | 4 | pairs per optimiser update (about 25 queries) | only with `pair_batch` |
| `hard_neg_weight` | 1.0 (off; `BASE` now: 3.0) | cells near the accepted targets of the other queries of the same pair count that many times in the softmax (`+log(weight)` on their logit) | to cut wrong_keypoint errors (left / right mix-ups); try 2-4; needs `pair_batch` |
| `hard_neg_radius` / `hard_neg_exclusion` | 1.0 / 2.0 cells | size of the negative zone around another query's target / zone around the query's own targets that is never negative | leave unless the paired wrong_keypoint rate suggests label noise |
| `train_mode` | `finetune` | `lora`: low-rank updates on the attention query / value of the last `lora_blocks` blocks, everything else frozen | when fine-tuning overfits; update 0 = frozen DINOv2 |
| `lora_rank` / `lora_alpha` / `lora_dropout` / `lora_blocks` / `lr_lora` | 8 / 16 / 0.05 / 6 / 1e-4 | LoRA size, scale (`lora_alpha / lora_rank`), dropout, last N blocks, peak lr | with `train_mode` `lora` |
| `ema_decay` | 0.0 (off; `BASE` now: 0.99) | moving average of the trained weights; validation and `best.pt` use the averaged weights | try 0.99 (memory of about 100 updates) for a steadier best epoch |
| `fusion` | `none` (`BASE` now: `concat`) | `concat`: `normalize(cat(dino, sd_concat_weight * sd))`, no fusion parameters: frozen gate in `evaluate`, or DINOv2 fine-tuned on the concatenated features in `train` (D-008). `adapter`: `normalize(dino + adapter(sd))`, last adapter layer starts at zero | `concat`: the G-SD gate, and the D-008 training run; `adapter`: stage 2, from a trained model, if the gate passes |
| `sd_model_id` / `sd_model_revision` | SD 2.1 community repo / pinned commit | which Stable Diffusion weights are loaded (about 5 GB on the first run, GPU) | with `fusion` other than `none` |
| `sd_image_size` / `sd_timestep` / `sd_up_block` | 736 / 261 / 1 | SD input side, noising step, decoder block of the feature map (46 cells = the DINOv2 grid at 644) | leave (DIFT recipe) |
| `sd_concat_weight` | 0.5 | weight of the SD part in `concat` | 0.3 / 1.0 if the gate is close |
| `adapter_layers` / `adapter_dim` / `lr_adapter` | 2 / 256 / 1e-4 (`BASE` now and `BASE` + `V2_TRAINING`: 3e-4) | adapter size and peak lr | with `fusion` `adapter` |

**Checkpoints.** LoRA and adapter weights are saved in `best.pt` like the unfrozen blocks (every parameter with
`requires_grad`); with `ema_decay` the file holds the averaged weights. `--run <run_id>` rebuilds the model from
the run's own `config.json` (LoRA, adapter, `fusion`), then loads `best.pt`. Stable Diffusion weights are never
saved: they are frozen and are downloaded again from the Hub at the pinned `sd_model_revision`. Runs made before
these keys existed still load, with the v3 switches off (`V3_OFF` in `src/config.py`), whatever `BASE` says at that moment.

**Deciding between runs.** Do not compare two validation scores by eye (the interval of one is about
+-4.5 points). `python scripts/compare_runs.py a_s0,a_s1 b_s0,b_s1` scores the two settings on the same queries
and bootstraps the validation pairs: difference, 95% interval, P(A better) for PCK@0.05/0.10/0.15, single / multi
and every error rate. Each run needs `per_query.csv`: run `python run.py --mode evaluate --run <run_id>` after
training, because training does not write it.

## 9. Reading the W&B charts

Charts are logged once per optimiser update (train) and once per epoch (val). System charts (GPU, CPU, disk)
and the console capture are off.

**Training (`train/`, `lr/`, `optim/`, `batch/`)**

| Chart | What it is | How to read it (run 1 values) |
|---|---|---|
| `train/loss_over_chance` | (loss − floor) / (chance − floor). chance = loss of a uniform p: log N for `soft_ce`, log(N / N_pos) for `marginal` (N = 2,116 cells, N_pos = positive cells). floor = lowest possible loss: the target entropy for `soft_ce`, 0 for `marginal` | 1 = chance, 0 = perfect, above 1 = confidently wrong (the frozen model at a low temperature). Replaces the raw `train/loss`: comparable between the two losses, temperatures and image sizes. Train has augmentation and label jitter, so it is not equal to the val value |
| `train/loss_over_chance_ema` | 0.95 EMA (memory ≈ 20 updates) | the trend; the per-update value is noisy |
| `train/loss_over_chance/{single, multi}` | the same for queries with 1 / several accepted targets | the floor already includes the extra ln K of multi targets, so the two are on one scale |
| `train/entropy_norm` (+ `_ema`) | H(p) / log N, entropy of the predicted distribution over the N cells. Replaces `train/softmax_entropy` (= entropy_norm × 7.66 nats at 644 px) | 0 = one-hot, 1 = uniform. For `soft_ce` with sigma 1 the ideal is ≈ 0.43–0.46 (3.3–3.5 nats); `marginal` can go lower. Run 1 in these units: 0.34 at start, peak 0.76 at the end of the warmup, 0.55 at the end. The peak is what the slower warmup should lower |
| `train/loss_temperature` | loss temperature of the update | 0.07 → 0.04 over epoch 0, then flat (only logged with `loss_temperature_start`) |
| `lr/block_10`, `lr/block_11`, `lr/layernorm` | learning rate per group, used by that update | OneCycleLR: rises over the first `warmup_epochs` (all of epoch 0), then cosine to ≈ 0 |
| `batch/effective` | samples per update | flat at 16 without `effective_batch_schedule` |
| `optim/grad_norm` | total gradient norm before clipping | large because the logits are divided by T = 0.03 (×33): not instability. Run 1: 27–44 in epoch 0, ≈ 11 after |
| `optim/grad_clipped` | 1 when the norm is above `grad_clip` (7) | should be mostly 1 during the warmup and less often after |

**Validation (`val/`, each epoch)**

| Chart | How to read it |
|---|---|
| `val/pck@0.1` | the competition metric on 453 queries; `best.pt` follows it |
| `val/pck@0.1_ci95` | half-width of the 95% interval (≈ ± 4.5). Optimistic (queries of a pair are correlated); for two settings on the same queries a smaller difference can still be real |
| `val/pck@0.05`, `val/pck@0.15` | precision (0.05) and coarse correctness (0.15) |
| `val/pck@0.1/{single, multi}` | multi-target queries score higher although their loss is higher |
| `val/pck@0.1_gain` | val PCK of this epoch minus the previous epoch (from epoch 1) |
| `val/pck@0.1_gain_over_ci95` | that gain divided by `val/pck@0.1_ci95`. Between −1 and 1 = the epoch's change is inside the noise of the validation set; when it stays there, more epochs are not buying a measurable gain. Same caveats as the interval |
| `val/loss_over_chance`, `val/loss_over_chance/{single, multi}`, `val/entropy_norm` | as the `train/` charts, without augmentation and label jitter, always at `loss_temperature`; `loss_over_chance` should go down with PCK |
| `val/flip_rate` | share of queries where `flip_test` picked the mirrored source (0 while it is off) |
| `val/error_types` (bar plot) | correct (d < 0.1 L); wrong_keypoint (on another labelled part of the same pair, e.g. left ↔ right eye); near_miss (0.1–0.2 L); other_miss (0.2–0.5 L); far_miss (> 0.5 L). Run 1: fine-tuning removed far / other misses; near_miss (16%) and wrong_keypoint (10%) are what is left |
| `val/per_group`, `val/dist_hist` | table per single/multi target and edge distance, histogram of d / L |
| `val/worst_predictions` (train only) | the `n_visual_checks` (4) largest errors, four panels per row: source with the query point | source CLS map | target with the accepted targets (green), the prediction (red ×) and a dashed line to the nearest accepted target | target CLS map. CLS map = cosine between the image's DINOv2 CLS token and each patch token (current weights), on the patch grid. Bright = the parts the global embedding describes, normally the object. A query or target point on a dark area = a part the global embedding ignores. Titles: d / L, error type and CLS cos(src, trg), the global similarity of the pair |
| `val/sweep` (evaluate only) | matching settings scored on the same features |

Run summary (not charts): `best_pck`, `micro_batch`, `total_updates`, `trainable_params_M`, `probe_peak_gb`,
`n_train_queries`.

## 10. Settings

- To change a setting, edit `BASE` and start a new run. The run's `config.json` records it.
- Paths depend on the runtime (detected automatically) and can be overridden with `SC_DATA_DIR`,
  `SC_OUTPUT_DIR`, `SC_WEIGHTS_DIR`, `SC_FEATURES_DIR`, `SC_SPLITS_DIR`.
- The model is `facebook/dinov2-base` at a pinned revision (`model_revision`). If `weights/dinov2-base`
  exists, it is loaded from there instead of the Hub.
- `cache_features_to_disk=True` (frozen mode, no augmentation) stores backbone features in `data/features/`.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `Missing .../train.csv` | `python scripts/kaggle_io.py download` (rules accepted? keys in `.env`?) |
| `Nothing to train: train_mode='frozen' with head='none'` | Use `train_mode="finetune"` (the default) |
| `Even a micro-batch of 1 does not fit the GPU` | Lower `image_size` or `unfreeze_blocks`, or set `batch_size` to an int |
| `fused attention kernels unavailable` | Keep `attn_implementation="sdpa"` |
| `image_size ... multiple of the patch size` | Use a multiple of 14: 448, 518, 644, 756, 840 |
| W&B asks to log in | Set `WANDB_API_KEY`, or `wandb_mode="offline"` / `"disabled"` in `BASE` |
| Colab lost the session | `RESUME = "<run_id>"` in the notebook and run again |
| `splits/val_pair_ids.json has pair ids that are not in train.csv` | The data changed: delete the file to rebuild the split, and tell the team |
