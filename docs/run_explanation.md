# Why the repo has so many "run" files

A first look at the repo shows `run.py`, `RUN.md`, `src/utils/run.py`, `colab/run.ipynb`,
`notebooks/kaggle_run.ipynb` and an `outputs/<run_id>/` folder for every command. They are not
copies of each other. Each one handles a different part of the same idea, the **run**.

> **A run** is one execution of one mode (`train`, `evaluate`, `submit`) with one
> config. It gets an id (`20261103-165509_evaluate`) and its own folder,
> `outputs/<run_id>/`. The folder holds the settings that produced the run and everything
> the run wrote.

The team works on three machines: a laptop, Colab (GPU, for training) and Kaggle (offline, for
the submission notebook). The project needs:

1. **One way to run the code** on all three, so a result does not depend on where it was
   produced.
2. **One record per run**, so any score can be traced back to its
   config, its code version and its checkpoints, even after a Colab disconnect.

The "run" files are what makes both of these possible.

## 1. The files

| File | Type | What it is for | Who uses it |
|---|---|---|---|
| `run.py` | Python CLI | **The only entry point.** It parses `--mode/--run/--resume`, checks the files and the device (`preflight`), creates the run folder, builds the model and calls `train`, `evaluate` or `submit` | People, the notebooks |
| `src/utils/run.py` | Library module | **Run folder logic:** `prepare_run_dir` creates or reopens `outputs/<run_id>/` and writes `config.json` once. `load_run_config` reloads a run's saved settings. `git_commit` records the code version (`-dirty` if there are uncommitted changes). `seed_everything` sets the seeds | `run.py`, `src/train.py` |
| `docs/RUN.md` | Guide | **How to run everything:** setup, the three modes, flags, local/Colab/Kaggle steps, troubleshooting | People |
| `colab/run.ipynb` | Notebook | **Thin Colab runner.** One cell sets `BRANCH/MODE/RUN/RESUME`. Then it calls `scripts/colab_setup.py` and `python run.py ...`. It holds no logic | Teammates training on a GPU |
| `notebooks/kaggle_run.ipynb` | Notebook | **Thin Kaggle runner (internet off).** It sets the weights folder and offline mode, then calls `python run.py --mode submit`. Kaggle uses the `submission.csv` it writes | Final submission |
| `outputs/<run_id>/` | Generated folder (not in git) | **The record of one run:** `config.json` (always), then `summary.json`/`per_query.csv`/`per_category.csv`/`visual_check.png`, or `checkpoints/`/`history.json`, or `predictions.csv`/`submission.csv` | Everyone; `--run` and `--resume` reopen it |

Supporting files with a related job:

| File | Role in a run |
|---|---|
| `scripts/colab_setup.py` | Gets a Colab session ready **before** `run.py` starts: pulls the branch, installs packages, loads secrets, mounts Drive, downloads data and weights. Its small `run()` function only runs shell commands and has nothing to do with run folders |
| `scripts/kaggle_io.py` | Downloads the competition data before the first run. Uploads a run's `submission.csv` after it |
| `scripts/download_weights.py` | Saves the DINOv2 weights locally, so runs can work offline |
| `src/config.py` | The only configuration (`BASE`) and the runtime paths. `output_dir` decides where run folders go |

## 2. How they interact

```text
          laptop                         Colab                             Kaggle (offline)
            │                              │                                    │
            │                    colab/run.ipynb                     notebooks/kaggle_run.ipynb
            │                    └─ scripts/colab_setup.py           └─ sets SC_WEIGHTS_DIR,
            │                       (pull, install, secrets,            HF_HUB_OFFLINE
            │                        Drive, data, weights)                    │
            ▼                              ▼                                    ▼
  ┌──────────────────────────── python run.py --mode ... ───────────────────────────────┐
  │ parse_args → load_config (src/config.py, or the saved config.json for --run/--resume) │
  │ preflight (files, GPU) → seed_everything → prepare_run_dir → build_model              │
  └─────────────┬───────────────┬────────────────┬─────────────────────────────────────┘
                ▼               ▼                ▼
          src/train.py    src/evaluate.py   src/predict.py
          (train)          + diagnostics    + write/check_submission (submit)
                └───────────────┴────────┬───────┘
                                         ▼
                              outputs/<run_id>/   (Drive on Colab, /kaggle/working on Kaggle)
                                         │
                 scripts/kaggle_io.py submit --file outputs/<run_id>/submission.csv
```

The notebooks and scripts only **prepare the machine**. `run.py` and `src/` do **the work**, and
it is the same code on every machine. Paths change with the runtime. `src/config.py` detects
the runtime (local, Colab or Kaggle) and resolves `data_dir`, `output_dir`, `weights_dir` and
`features_dir`.

## 3. What happens to a run folder

| Command | Effect on `outputs/<run_id>/` |
|---|---|
| `run.py --mode X` | Creates a **new** folder `<UTC time>_<mode>` and writes `config.json` with the full config and `git_commit` |
| `run.py --mode train --resume <run_id>` | **Reopens** the folder and continues from `checkpoints/last.pt` (weights, optimiser, scheduler, AMP scaler, epoch, batch, random state) |
| `run.py --mode evaluate\|submit --run <run_id>` | **Reopens** the folder, loads `checkpoints/best.pt`, and writes the evaluation files, or `predictions.csv` and `submission.csv`, into it |

Rules that keep runs comparable:

- `config.json` is written **once**, when the run is created. Reopening a run never overwrites
  it, so the file always records how the run started.
- `--run` and `--resume` use the run's saved `config.json`, not the current `BASE`. A change of
  settings needs a new run.
- When a run is reopened on a different commit, `run.py` prints the old and the new commit.
- Seeds and `cudnn.deterministic` are set before every run, so the same config gives the same
  numbers.