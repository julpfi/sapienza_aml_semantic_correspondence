# Semantic correspondence: AML Challenge 2026/27

Our code for the [AML Challenge](https://www.kaggle.com/competitions/aml-challenge-2026-27/overview):
given a source image, a target image and a source point, predict the corresponding point in the
target image. The score is PCK@0.10.

## Structure

```text
colab/run.ipynb     Thin Colab runner: secrets, clone/pull, setup, run
scripts/            setup_colab.py and run_colab.py
splits/             Shared validation split, tracked in git
notebooks/          Data exploration
src/config.py       Default settings and named experiments
src/data_loader/    Datasets and transforms
src/models/         Backbones and matching methods
src/utils/          Paths, run folders and checkpoints
src/train.py        Training entry point
src/evaluate.py     Validation predictions and PCK
src/diagnostics.py  Plots and failure analysis
src/predict.py      Submission file
soco/, run.py       SOCO DINO-first pipeline (Lang branch), see .claude/pipeline.md
data/               Competition data (not in git)
outputs/<run_id>/   Config, checkpoints, predictions and submission (not in git)
```

## Run on Colab

1. In Colab, go to File > Open notebook > GitHub, search for this repo and open `colab/run.ipynb`.
   Do not clone manually: the notebook does this.
2. Select a GPU under Runtime > Change runtime type.
3. First time only: accept the competition rules on Kaggle and request DINOv3 access on Hugging Face.
4. First time only: add `KAGGLE_USERNAME`, `KAGGLE_KEY`, `WANDB_API_KEY` and `HF_TOKEN` to Colab
   Secrets and allow notebook access.
5. Set the options in the first code cell and run all cells.

Commit and push changes before running them, because Colab pulls code from GitHub. Runs are saved
to each person's own Google Drive, under `aml_semantic_correspondence/outputs`.

## Checkpoints

- `last.pt` and `best.pt` are saved to the run folder on your own Drive. To continue after a
  disconnect, set `RESUME` in the notebook to the run ID and leave `SET` empty. The run reuses the
  settings in its `config.json`.
- At the end of training, `best.pt` is uploaded to W&B as the artifact `<run_id>-best`.
- To start from a teammate's model, copy the artifact's full name from W&B and use
  `--set init_artifact=<entity>/<project>/<run_id>-best:latest`.

## Run locally

Use a Miniconda environment called `advance`. Create it once:

```bash
conda env create -f environment.yml
conda activate advance
# PyTorch for your machine (https://pytorch.org/get-started/locally/), for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU only
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # NVIDIA GPU
```

Then, in every new terminal:

```bash
conda activate advance
python -m src.train --exp baseline --dry-run
```

After `requirements.txt` changes, update the environment with
`conda env update -n advance -f environment.yml`.

## Add an experiment

Add an entry to `EXPERIMENTS` in `src/config.py`, listing only what differs from `BASE`. For
one-off changes, use `--set`, for example `--set lr=1e-4 batch_size=8`.
