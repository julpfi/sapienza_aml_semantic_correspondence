# RUN.md: how to run the project

How to set up the project, run every step and submit, locally, in Colab or on Kaggle; how the "run" files fit
together; the fine-tuning hyperparameters; and how to read the W&B charts.
The analysis of the first fine-tuning run is in [REVIEW_finetuning.md](REVIEW_finetuning.md).

## 1. What the pipeline does

For each query (source image, target image, point on the source), it predicts the matching
point on the target. The score is PCK@0.10 (see `.claude/CLAUDE.md`).

```text
image ──► square resize 644 px ──► DINOv2-B/14 (facebook/dinov2-base) ──► 46×46 patch features
      ──► head (linear, identity at start) ──► cosine similarity of the source point vs every target cell
      ──► 2× upsampling + 11×11 soft-argmax (T = 0.02) ──► (x, y) in original target pixels
```

There is one configuration: `BASE` in `src/config.py`. It is set up for fine-tuning on a Colab T4
(`head="linear"`, `train_mode="finetune"`, last DINOv2 block). Before training, the linear head is the
identity, so `evaluate`/`submit` without `--run` give the frozen DINOv2 baseline (val PCK@0.10 = 40.4 with
the old 5×5 window, 41.1 with the 11×11 window, checked 2026-11-05). A trained model is used with `--run <run_id>`.
Hyperparameters: section 8; design reasons: [PRD_finetuning.md](PRD_finetuning.md) section 6.

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
    per_group.csv      PCK per single/multi target and per edge distance
    error_types.csv    correct / near_miss / wrong_keypoint / far_miss / other_miss
    sweep.csv          flip_test x soft_window x soft_temperature (x image_size x model): a report only
    visual_check.png   the worst validation predictions
    wandb/             local W&B files (charts are online at wandb.ai)
  20261104-130000_train/
    config.json
    history.json       validation PCK, loss and error mix after every epoch
    checkpoints/       last.pt (resume), best.pt (best validation PCK)
  20261104-140000_submit/
    config.json
    predictions.csv    id,x,y
    submission.csv     id,prediction (the Kaggle file)
```

| Command | What it does |
|---|---|
| `python run.py --mode evaluate` | Validation PCK, groups, error types, sweep, worst-case visual check, W&B |
| `python run.py --mode train` | Train with `BASE` (validation every epoch, best kept) |
| `python run.py --mode train --resume <run_id>` | **Reopen** the folder and continue from `last.pt` (weights, optimiser, scheduler, AMP scaler, epoch and batch position, random state) |
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
2. **Look at the failures:** `visual_check.png` and `per_query.csv` (`dist` = error / L).
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
once, in order, with finite values, and clips points to the image. Still open `visual_check.png` after a change
to resizing or coordinates.

## 6. Colab (training)

Open `colab/run.ipynb` in Colab (File > Open notebook > GitHub, this repo).

1. Runtime > Change runtime type > **GPU**.
2. Colab Secrets (key icon), with notebook access: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `WANDB_API_KEY`, `HF_TOKEN`.
3. In the settings cell, set `BRANCH`, `MODE` (and `RUN` or `RESUME`), then run all cells. The run uses `BASE`
   from the pushed branch.

- `%run scripts/colab_setup.py` pulls the branch, pip-installs `environment.yml` (keeping Colab's CUDA torch),
  loads the secrets, mounts Drive, downloads the data to `/content/data/raw` and saves the weights to Drive.
- Runs are saved to `MyDrive/aml_semantic_correspondence/outputs/<run_id>`, so they survive a disconnect.
- **After a disconnect:** set `RESUME = "<run_id>"` and run all cells again. `last.pt` is also saved every
  `checkpoint_every_steps` updates (150).
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

Current `BASE` (run 2) next to run 1 (`20261104-155818_train`, val PCK@0.10 62.9, 66.0 with the new
inference window). The numbers in the last column are sections of [REVIEW_finetuning.md](REVIEW_finetuning.md).
The optimiser settings (learning rates, schedule, clip) were chosen afterwards and replace the ones in its 2.4.

| Setting | Now | Run 1 | Why |
|---|---|---|---|
| **What is trained** | | | |
| `train_mode` / `head` | finetune / linear | same | head initialised as the identity: update 0 = frozen DINOv2 (1.2) |
| `unfreeze_blocks` | **1** (block 11 + final LayerNorm + head, 7.68M params) | 2 (14.77M) | fewer weights moved on 2,042 queries; less forgetting |
| **Optimiser** | AdamW + OneCycleLR | AdamW + warmup/cosine | |
| `lr_head` (peak) | **1e-3** | 1e-4 | 100× the backbone: the identity head is the cheap part to move. Watch the entropy and `embed_var_cells` (collapse) |
| `lr_backbone` (peak) | 1e-5 (block 11 and final LayerNorm) | same | |
| `layer_decay` | 0.75 | same | lr × 0.75 per block below the top; no effect with 1 block |
| `weight_decay` | 0.05, **on every trained parameter** | 0.05, not on LayerNorm / biases | plain AdamW; negligible at these learning rates (1.10) |
| **Schedule** | | | |
| `epochs` / `early_stop_patience` | **5 / 2** | 7 / 3 | run 1 plateaued after epoch 3 (1.15) |
| `effective_batch` | 16 (micro-batch auto) | same | 127 updates per epoch, 635 in total |
| `lr_pct_start` | **0.05** (≈ 32 updates) | warmup 10% | OneCycleLR: lr rises from peak / 25 to the peak, then cosine to peak / 25e4. Other arguments at their defaults, so AdamW's beta1 also cycles 0.95 → 0.85 → 0.95 |
| `grad_clip` | **5** | 1 | the normal norm is ≈ 11 (run 1), so most updates are clipped: `grad_clipped` ≈ 1 is expected (1.9) |
| `amp` | fp16 + GradScaler | same | the T4 has no bf16 |
| **Loss** | | | |
| `loss_temperature` | 0.03 | same | |
| `label_sigma` | 1.0 cell | same | |
| `label_jitter` | **0.25 cell** | 0.5 | less label blur, for small objects (1.12) |
| **Augmentation** (very light) | | | |
| `flip_augment_prob` | 0.5 (both images) | same | keeps left/right consistent |
| `aug_image_prob` (new) | **0.10** | (all images) | only 10% of the images go through the steps below: 9.7% measured change, run 1: 96% (1.14) |
| `aug_crop_scale` / prob | **[0.9, 1.0]** / 0.5 | [0.7, 1.0] / 0.5 | a crop removes at most 10% of the side |
| `aug_rotation_deg` / prob | **10°** / 0.5 | 15° / 0.5 | |
| `aug_color_jitter` / `aug_hue_jitter` / prob | **0.2 / 0.02** / 0.8 | 0.3 / 0.05 / 0.8 | |
| `aug_grayscale_prob` / `aug_blur_prob` | **0.05 / 0.05** | 0.1 / 0.1 | |
| **Inference** (validation each epoch and submission) | | | |
| `soft_window` / `soft_temperature` | **11 / 0.02** | 5 / 0.04 | +3.1 PCK on run 1's `best.pt`, small objects 54.0 → 61.8 (2.1) |
| `flip_test` | False | same | +0.7 only after fine-tuning, inside the interval |

Run 2 changes several settings at once. If it is worse than run 1 by more than the interval (± 4.5), undo the
head learning rate first (1e-3 → 1e-4), then the augmentation. Compare with run 1 at the same inference settings (66.0).

## 9. Reading the W&B charts

Charts are logged once per optimiser update (train) and once per epoch (val). System charts (GPU, CPU, disk)
and the console capture are off.

**Training (`train/`, `lr/`, `optim/`, `batch/`)**

| Chart | What it is | How to read it (run 1 values) |
|---|---|---|
| `train/loss` | cross-entropy between the predicted cell distribution and the Gaussian bumps on the accepted targets, mean of the update | can never reach 0: its floor is the target entropy H(t) ≈ 3.3–3.5 nats. Run 1: 9 → 4.2. Not comparable with `val/loss` (train has augmentation and label jitter) |
| `train/loss_ema` | 0.95 EMA of `train/loss` (memory ≈ 20 updates) | the trend; the raw loss is noisier (std 0.17) than an epoch of progress |
| `train/loss/single`, `/multi` | loss of queries with 1 / several accepted targets | compare each only with itself over time: multi has a higher floor (+ ln K) |
| `train/softmax_entropy` (+ `_ema`) | entropy H(p) of the predicted distribution; e^H = effective number of cells | ideal ≈ 3.3–3.5 (27–33 cells). Below = overconfident, above = too spread. Run 1: 2.6 at start (confidently wrong), peak 5.8 at the end of the warmup, 4.2 at the end. Run 2 should show a smaller peak |
| `train/peak_mass` | probability inside a 5 × 5-cell window around the most likely cell | ≈ 1: one peak (maybe wide). Low: mass on several separate peaks (e.g. left and right eye). Tells apart what entropy alone cannot |
| `train/embed_var_batch` | 1 − ‖mean‖² of the batch's source descriptors | stays near its update-0 value = healthy; a steady fall = different queries get the same descriptor (collapse) |
| `train/embed_var_cells` | 1 − ‖mean‖² over the cells of each target map | a steady fall together with a rising entropy = the map is flattening (collapse): lower `lr_head` or stop |
| `lr/head`, `lr/block_11`, `lr/layernorm` | learning rate per group, used by that update | OneCycleLR: rises over the first 5% of the updates, then cosine to ≈ 0 |
| `batch/effective` | samples per update | flat at 16 without `effective_batch_schedule` |
| `optim/grad_norm` | total gradient norm before clipping (AMP scale removed) | large because the logits are divided by T = 0.03 (×33): not instability. Run 1: 27–44 in epoch 0, ≈ 11 after |
| `optim/grad_clipped` | 1 when the norm is above `grad_clip` | with clip 5 and a normal norm ≈ 11 it is mostly 1; read `grad_norm` for spikes instead |
| `optim/amp_scale` | fp16 GradScaler scale | drops in the first updates (fp16 overflows, updates skipped): normal |

**Validation (`val/`, each epoch)**

| Chart | How to read it |
|---|---|
| `val/pck@0.1` | the competition metric on 453 queries; `best.pt` follows it |
| `val/pck@0.1_ci95` | half-width of the 95% interval (≈ ± 4.5). Optimistic (queries of a pair are correlated); for two settings on the same queries a smaller difference can still be real |
| `val/pck@0.05`, `val/pck@0.15` | precision (0.05) and coarse correctness (0.15) |
| `val/pck@0.1/{single, multi}` | multi-target queries score higher although their loss is higher |
| `val/loss`, `val/loss/{single, multi}` | same loss without augmentation; should go down with PCK |
| `val/flip_rate` | share of queries where `flip_test` picked the mirrored source (0 while it is off) |
| `val/error_types` (bar plot) | correct (d < 0.1 L); wrong_keypoint (on another labelled part of the same pair, e.g. left ↔ right eye); near_miss (0.1–0.2 L); other_miss (0.2–0.5 L); far_miss (> 0.5 L). Run 1: fine-tuning removed far / other misses; near_miss (16%) and wrong_keypoint (10%) are what is left |
| `val/per_group`, `val/dist_hist`, `val/worst_predictions` | table per single/multi target and edge distance, histogram of d / L, the worst 8 queries drawn |
| `val/sweep` (evaluate only) | matching settings scored on the same features |

Run summary (not charts): `best_pck`, `val/n_queries`, `val/n_pairs`, `micro_batch`, `total_updates`,
`trainable_params_M`, `probe_peak_gb`, `n_train_queries`.

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
| `Nothing to train: head='none'` | Set `head="linear"` in `BASE` (it is the default) |
| `Even a micro-batch of 1 does not fit the GPU` | Lower `image_size` or `unfreeze_blocks`, or set `batch_size` to an int |
| `fused attention kernels unavailable` | Keep `attn_implementation="sdpa"` and fp16 (`amp=True`) |
| `image_size ... multiple of the patch size` | Use a multiple of 14: 448, 518, 644, 756, 840 |
| W&B asks to log in | Set `WANDB_API_KEY`, or `wandb_mode="offline"` / `"disabled"` in `BASE` |
| Colab lost the session | `RESUME = "<run_id>"` in the notebook and run again |
| `splits/val_pair_ids.json has pair ids that are not in train.csv` | The data changed: delete the file to rebuild the split, and tell the team |
