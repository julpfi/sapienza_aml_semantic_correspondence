# Semantic correspondence: AML Challenge 2026/27

Our code for the [AML Challenge](https://www.kaggle.com/competitions/aml-challenge-2026-27/overview):
given a source image, a target image and a source point, predict the corresponding point in the
target image. The score is PCK@0.10.

**How to run everything: [RUN.md](RUN.md).** What changed in the reorganization:
[docs/REORGANIZATION.md](docs/REORGANIZATION.md). Decisions: [docs/decisions.md](docs/decisions.md).

## Structure

```text
run.py              Single entry point: --mode train | evaluate | submit
environment.yml     Dependencies (conda env `advance`)
.env.example        Secret names; copy to .env (ignored by git)
src/config.py       The only configuration (BASE) and runtime paths (local / Colab / Kaggle)
src/models/         DINOv2 backbone (Hugging Face), head, matching, loss
src/train.py        Training: frozen backbone + head, or fine-tuning (Phase 2)
src/evaluate.py     Validation PCK
src/predict.py      Test predictions and submission file
src/diagnostics.py  Error breakdown, visual check
src/metrics.py      PCK@0.10
src/utils/          Run folders, checkpoints, secrets, feature cache
data/               Dataset code (+ data/raw/ for the download, not in git); see data/README.md
splits/             Shared validation split, tracked in git
scripts/            Colab setup, weight download, Kaggle download/submit
colab/run.ipynb     Thin Colab runner
notebooks/          Thin Kaggle runner (offline)
docs/               Reorganization guide, decision log, Lang reference results
outputs/<run_id>/   One folder per run: config.json + results (not in git), see RUN.md
```

## Quick start (local)

```bash
conda env create -f environment.yml
conda activate advance
cp .env.example .env                                 # add your Kaggle / W&B / HF keys
python scripts/kaggle_io.py download                 # competition data -> data/raw/
python run.py --mode evaluate                       # validation PCK@0.10 (Lang baseline: 40.4)
python run.py --mode submit                         # outputs/<run_id>/submission.csv
```

## Run on Colab

1. In Colab, go to File > Open notebook > GitHub, search for this repo and open `colab/run.ipynb`.
   Do not clone manually: the notebook does this.
2. Select a GPU under Runtime > Change runtime type.
3. First time only: accept the competition rules on Kaggle.
4. First time only: add `KAGGLE_USERNAME`, `KAGGLE_KEY`, `WANDB_API_KEY` and `HF_TOKEN` to Colab
   Secrets and allow notebook access.
5. Set the options in the first code cell and run all cells.

Commit and push changes before running them, because Colab pulls code from GitHub. Runs are saved
to each person's own Google Drive, under `aml_semantic_correspondence/outputs`.

## Settings

There is one configuration: `BASE` in `src/config.py`. To change a setting (for example
`head="linear"` to train a head), edit it there and start a new run. Every run saves the full
settings it used in `outputs/<run_id>/config.json`.

## Checkpoints

- `checkpoints/last.pt` and `checkpoints/best.pt` are saved to the run folder (your Drive on Colab).
  To continue after a disconnect, set `RESUME` in the notebook to the run ID, or run
  `python run.py --mode train --resume <run_id>`. The run reuses the settings in its `config.json`.
- At the end of training, `best.pt` is uploaded to W&B as the artifact `<run_id>-best`.
- To start from a teammate's model, copy the artifact's full name from W&B into `init_artifact`
  in `BASE` (`<entity>/<project>/<run_id>-best:latest`).
