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

There is one configuration: `BASE` in `src/config.py`. As it stands it is the Lang pipeline
(nothing trained, val PCK@0.10 = 40.4). To train, change these settings in `BASE`:

| Settings in `BASE` | What is trained | Where to run |
|---|---|---|
| defaults (`head="none"`) | nothing (Lang baseline) | anywhere, CPU is fine |
| `head="linear"` (or `"mlp"`) | a small head on frozen features (Phase 1) | GPU recommended |
| `head="linear"`, `train_mode="finetune"` | head + last 2 DINOv2 blocks (Phase 2) | Colab GPU |

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
    summary.json       PCK@0.05/0.10/0.15, score, counts
    per_query.csv      one row per validation query (prediction, dist = error / L)
    per_category.csv   PCK per category
    visual_check.png   a few predictions drawn on the images
  20261104-130000_train/
    config.json
    history.json       validation PCK after every epoch
    checkpoints/       last.pt (resume), best.pt (best validation PCK)
  20261104-140000_submit/
    config.json
    predictions.csv    id,x,y
    submission.csv     id,prediction (the Kaggle file)
```

| Command | What it does | Writes in `outputs/<run_id>/` |
|---|---|---|
| `python run.py --mode evaluate` | Validation PCK on the held-out pairs, error breakdown, visual check | `summary.json`, `per_query.csv`, `per_category.csv`, `visual_check.png` |
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
4. **Train** when the training-free result is not enough: set `head="linear"` in `BASE`, then
   `python run.py --mode train`, and on Colab also `train_mode="finetune"`. Evaluate and submit the
   trained run with `--run <run_id>`.

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
  state). For long epochs, set `checkpoint_every_steps=200` in `BASE` before starting.
- The last cell can upload a `submission.csv` to Kaggle.

Phase 2 settings (`head="linear"`, `train_mode="finetune"`, all in `src/config.py`): `unfreeze_blocks=2`,
`lr_backbone=1e-5` and `lr_head=1e-4` (two parameter groups), AdamW wd 0.05, OneCycle with 5% warmup,
fp16 AMP, `grad_accum_steps=1`, `batch_size=4`, `epochs=2`. The frozen blocks stay in eval mode.

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
| `Nothing to train: head='none'` | The baseline has nothing to train: set `head="linear"` in `BASE` (and `train_mode="finetune"` for Phase 2) |
| `image_size ... multiple of the patch size` | Use a multiple of 14: 448, 518, 644, 756, 840 |
| W&B asks to log in | Set `WANDB_API_KEY`, or `wandb_mode="offline"` / `"disabled"` in `BASE` |
| Colab lost the session | `RESUME = "<run_id>"` in the notebook and run again |
| `splits/val_pair_ids.json has pair ids that are not in train.csv` | The data changed: delete the file to rebuild the split, and tell the team |
