# RUN.md: how to run the project

How to set up the project, run every step, and submit, locally, in Colab or on Kaggle.
The repo layout and what moved where are in [docs/REORGANIZATION.md](docs/REORGANIZATION.md).

## 1. What the pipeline does

For each query (source image, target image, point on the source), it predicts the matching
point on the target. The score is PCK@0.10 (see `.claude/CLAUDE.md`).

```text
image ──► square resize 644 px ──► DINOv2-B/14 (frozen, facebook/dinov2-base) ──► 46×46 patch features
      ──► head (identity by default) ──► cosine similarity of the source point vs every target cell
      ──► 2× upsampling + 5×5 soft-argmax ──► (x, y) in original target pixels
```

There is one configuration: `BASE` in `src/config.py`. It is set up for fine-tuning on a Colab T4
(`head="linear"`, `train_mode="finetune"`, last 2 DINOv2 blocks). Before training, the linear head is the
identity, so `evaluate`/`submit` without `--run` give the Lang baseline (val PCK@0.10 = 40.4). A trained
model is used with `--run <run_id>`. Every training choice and its reason: [PRD_finetuning.md](PRD_finetuning.md) section 6.

| What | Where to run |
|---|---|
| `evaluate` / `submit` (baseline or `--run`) | anywhere; CPU works but takes ~1 h for validation |
| `train` | Colab T4 (`--require-gpu` in the notebook) |

Everything goes through one command: `python run.py --mode train|evaluate|submit`.

## 2. One-time setup (local)

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

Check the setup without running anything:

```bash
python run.py --mode submit --dry-run
```

## 3. The three modes and the output folder

Every command creates (or reopens) one run folder `outputs/<run_id>/`. `<run_id>` is
`<UTC time>_<mode>`, for example `20261104-120000_evaluate`, and each command prints it.
`config.json` (every setting the run used, plus the git commit) is always the first file written.

```text
outputs/
  20261104-120000_evaluate/
    config.json        settings of this run
    summary.json       PCK@0.05/0.10/0.15 (+95% interval), score, error mix, PCK and loss per group
    per_query.csv      one row per validation query (prediction, dist = error / L, groups, error type)
    per_group.csv      PCK per object size (small/medium/large), single/multi target, edge distance
    error_types.csv    correct / near_miss / wrong_keypoint / far_miss / other_miss
    sweep.csv          flip_test x soft_window x soft_temperature (x image_size x model): a report only
    visual_check.png   the worst validation predictions
    wandb/             local W&B files (charts are online at wandb.ai)
  20261104-130000_train/
    config.json
    history.json       validation PCK, loss and error mix after every epoch
    checkpoints/       last.pt (resume), best.pt (best validation PCK)
    wandb/
  20261104-140000_submit/
    config.json
    predictions.csv    id,x,y
    submission.csv     id,prediction (the Kaggle file)
```

| Command | What it does | Writes in `outputs/<run_id>/` |
|---|---|---|
| `python run.py --mode evaluate` | Validation PCK on the held-out pairs, groups, error types, sweep, worst-case visual check, W&B | `summary.json`, `per_query.csv`, `per_group.csv`, `error_types.csv`, `sweep.csv`, `visual_check.png` |
| `python run.py --mode train` | Train with the `BASE` settings (validation every epoch, best kept) | `history.json`, `checkpoints/last.pt`, `checkpoints/best.pt` |
| `python run.py --mode train --resume <run_id>` | Continue a training run after a disconnect | same folder |
| `python run.py --mode evaluate --run <run_id>` | Evaluate a trained run with its `best.pt` | evaluation files, in that run's folder |
| `python run.py --mode submit` | Test predictions + Kaggle file, checked against `sample_submission.csv` | `predictions.csv`, `submission.csv` |
| `python run.py --mode submit --run <run_id>` | Same, for a trained run | in that run's folder |

Other flags: `--require-gpu` stops if there is no CUDA GPU; `--dry-run` only prints the settings
and checks the files.

## 4. Typical workflow

1. **Validate:** `python run.py --mode evaluate`. Expect PCK@0.10 ≈ 40.4 on 453 queries.
   The first run writes `splits/val_pair_ids.json`; commit it.
2. **Look at the failures:** `visual_check.png` and `per_query.csv` (`dist` = error / L).
3. **Submit:**
   ```bash
   python run.py --mode submit                    # or --run <run_id> for a trained model
   python scripts/kaggle_io.py submit --file outputs/<run_id>/submission.csv --message "baseline: val 40.4"
   ```
4. **Train** on Colab (section 5) with `MODE = "train"`. Evaluate and submit the trained run with
   `--run <run_id>`. The order of runs and what to change between them: PRD_finetuning.md section 6.11.
5. **Sweep:** `evaluate` writes `sweep.csv`. Copy a better setting into `BASE` by hand only if its gain
   is larger than its 95% interval (`ci95`). The submission never reads `sweep.csv`.
   For resolutions/ViT-L, fill `sweep_image_size` / `sweep_model_id` in `BASE` and run evaluate on the T4.
6. **Check the augmentation** after changing an `aug_*` setting: `python scripts/check_augmentation.py`
   (draws augmented pairs, and checks that no point is moved next to the border).

Before every submission (CLAUDE.md): `submit` already checks that every id of
`sample_submission.csv` appears exactly once, in order, with finite values, and it clips
points to the image. Still open `visual_check.png` after a change to resizing or coordinates.

## 5. Colab (training, Phase 2)

Open `colab/run.ipynb` in Colab (File > Open notebook > GitHub, this repo). It only sets options
and calls the repo code.

1. Runtime > Change runtime type > **GPU**.
2. Colab Secrets (key icon), with notebook access allowed: `KAGGLE_USERNAME`, `KAGGLE_KEY`,
   `WANDB_API_KEY`, `HF_TOKEN`.
3. In the settings cell, set `BRANCH`, `MODE` (and `RUN` or `RESUME`), then run all cells.
   The run uses `BASE` from the pushed branch.

What happens:
- `%run scripts/colab_setup.py` pulls the branch and pip-installs `environment.yml`, keeping
  Colab's own CUDA torch. It also loads the secrets, mounts Drive, downloads the data to
  `/content/data/raw`, and saves the weights to Drive.
- Then `python run.py ...` runs. Runs are saved to
  `MyDrive/aml_semantic_correspondence/outputs/<run_id>`, so they survive a disconnect.
- **After a disconnect:** set `RESUME = "<run_id>"` and run all cells again. Training continues
  from `last.pt` (weights, optimiser, scheduler, AMP scaler, epoch and batch position, random
  state). `last.pt` is also saved every `checkpoint_every_steps` updates (150 by default).
- The last cell can upload a `submission.csv` to Kaggle.

Training settings (all in `BASE`, reasons in PRD_finetuning.md section 6): last 2 blocks + final LayerNorm +
identity-initialised linear head; `lr_backbone=1e-5` for the top block with `layer_decay=0.75` per block below,
`lr_head=1e-4`; linear warmup (10% of updates) then cosine to 1%, set every update; AdamW, weight decay 0.05
(not on LayerNorm/bias); fp16 AMP (the T4 has no bf16); `effective_batch=16`, with the micro-batch found by a
GPU memory probe at start-up (`batch_size="auto"`) and gradient accumulation for the rest; 7 epochs with early
stop after 3 epochs without a better val PCK; flip, crop, rotation, colour, grayscale and blur augmentation.
The W&B run shows loss (overall, per object size, single/multi target), learning rate per block, batch sizes,
gradient norm, AMP scale, softmax entropy, GPU memory, and every validation metric per epoch.

Push your code to the branch before running it: Colab pulls from GitHub, and edits made in Colab are lost.

## 6. Kaggle (offline submission notebook)

1. Locally: `python scripts/download_weights.py`, then upload the `weights/` folder as a Kaggle
   dataset (it must contain `dinov2-base/`).
2. Upload the repo as a Kaggle dataset, without `data/raw`, `weights/` or `outputs/`.
3. Create a notebook from `notebooks/kaggle_run.ipynb`, attach the competition data, the repo
   dataset and the weights dataset, and turn the internet off.
4. Set `REPO_DIR` and `WEIGHTS_DIR` in the first code cell and run all. `--mode submit` writes
   `/kaggle/working/submission.csv`, which Kaggle uses as the submission.

## 7. Settings

All settings are in `BASE` in `src/config.py`, with a comment on where each value comes from:
- To change a setting, edit `BASE` and start a new run. The run's `config.json` records it.
- `--run`/`--resume` reuse the saved `config.json` of that run, not the current `BASE`.
- Paths depend on the runtime (local, Colab or Kaggle, detected automatically) and can be overridden
  with `SC_DATA_DIR`, `SC_OUTPUT_DIR`, `SC_WEIGHTS_DIR`, `SC_FEATURES_DIR`, `SC_SPLITS_DIR`.
- The model is `facebook/dinov2-base` at a pinned revision (`model_revision`). If `weights/dinov2-base`
  exists, it is loaded from there instead of the Hub.
- `cache_features_to_disk=True` (frozen mode only) stores backbone features in `data/features/`,
  so later head trainings skip DINOv2.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Missing .../train.csv` | `python scripts/kaggle_io.py download` (rules accepted? keys in `.env`?) |
| `Nothing to train: head='none'` | Set `head="linear"` in `BASE` (it is the default) |
| `Even a micro-batch of 1 does not fit the GPU` | Lower `image_size` or `unfreeze_blocks`, or set `batch_size` to an int |
| `fused attention kernels unavailable` | Attention falls back to a slower, memory-hungry kernel: keep `attn_implementation="sdpa"` and fp16 (`amp=True`) |
| `image_size ... multiple of the patch size` | Use a multiple of 14: 448, 518, 644, 756, 840 |
| W&B asks to log in | Set `WANDB_API_KEY`, or `wandb_mode="offline"` / `"disabled"` in `BASE` |
| Colab lost the session | `RESUME = "<run_id>"` in the notebook and run again |
| `splits/val_pair_ids.json has pair ids that are not in train.csv` | The data changed: delete the file to rebuild the split, and tell the team |
