# PRD: reorganization of the merged repository

Status: **draft, waiting for approval**. No other file is changed until this is approved.
Date: 2026-11-03.

## 1. Problem and goal

The repository combines two codebases that were never integrated:

- **main** has the reference *structure*: `src/` package, `config.py` with `BASE` + `EXPERIMENTS`
  + `--set`, run folders, checkpoints, W&B, Colab runner. It has **no working pipeline**:
  `train.py` stops with `NotImplementedError`, and `evaluate.py`, `predict.py`, `diagnostics.py`,
  `data_loader/` and `models/` hold only TODO docstrings.
- **Lang** (`Lang_Pipeline_Kaggle_Submission`, commit `0616d1f`) has the reference *logic*: the
  `soco/` package, `run.py`, and `notebooks/soco_kaggle.ipynb`. They produced the first Kaggle
  submission (val PCK@0.10 = **40.4%** on 453 queries).

The code that runs on the competition data lives partly in **notebook cells**
(`AMLQueryDataset`, `AMLPCKMeter`, the validation split, the test-CSV adapter), not in the package.

**Done means:**

1. There is one codebase in main's layout, and every piece of logic exists in exactly one place.
2. `python run.py --mode submit` reproduces the Lang submission. Format must be identical; values
   must be equivalent (section 10).
3. Every setting is in `src/config.py`, and runtime paths (local / Colab / Kaggle) are detected
   automatically.
4. DINOv2 is loaded from Hugging Face `facebook/dinov2-base` with a pinned revision.
5. Phase 2 (fine-tuning in Colab) is ready to launch from `colab/run.ipynb` but has not been run.
6. `environment.yml` is the only dependency file, and `.env.example` documents the secrets.

## 2. Current state

### 2.1 File map

| File | Origin | What it does | Depends on |
|---|---|---|---|
| `README.md` | main | Project overview, Colab and local instructions | — |
| `src/config.py` | main | `BASE` dict (seed 1019, W&B, `model`, `image_size` 512, batch 8, lr 1e-4, epochs 10), `EXPERIMENTS`, `parse_overrides`, `get_config` | — |
| `src/utils/paths.py` | main | `REPO_ROOT`, `DATA_DIR`, `RAW_DATA_DIR`, `SPLITS_DIR`, `OUTPUT_DIR` (env `SC_DATA_DIR`, `SC_OUTPUT_DIR`) | os |
| `src/utils/run.py` | main | `git_commit`, `load_run_config`, `prepare_run_dir`, `seed_everything` | config, paths, numpy, torch |
| `src/utils/checkpoint.py` | main | `save_checkpoint` (atomic), `load_checkpoint`, `publish_best` / `download_artifact` (W&B) | torch, wandb, paths |
| `src/train.py` | main | CLI `--exp/--set/--resume/--dry-run`, run folder, W&B init. **Loop is a TODO** | config, utils, wandb |
| `src/evaluate.py` | main | **TODO stub** (docstring: PCK@0.10, `scale` column = L) | — |
| `src/predict.py` | main | **TODO stub** | — |
| `src/diagnostics.py` | main | **TODO stub** | — |
| `src/data_loader/__init__.py` | main | **Empty** (docstring only) | — |
| `src/models/__init__.py` | main | **Empty** (docstring only) | — |
| `scripts/setup_colab.py` | main | `pip install -r requirements.txt`, Kaggle download + unzip into `data/raw`, report official files | config, paths, kaggle CLI |
| `scripts/run_colab.py` | main | GPU / secret checks, then `python -m src.train` | — |
| `colab/run.ipynb` | main | Secrets → env, Drive mount, clone/pull, `setup_colab.py`, `run_colab.py` | scripts |
| `splits/.gitkeep`, `data/.gitkeep`, `notebooks/.gitkeep` | main | Placeholders | — |
| `run.py` | Lang | CLI `eval/sweep/featsweep/decide/finetune/predict` + `--set` | soco |
| `soco/config.py` | Lang | `Config` dataclass (≈35 fields), `save/load/replace` | — |
| `soco/data.py` | Lang | `SPairDataset`, `CSVPairDataset`, `build_dataset`, `collate_keep` | numpy, PIL, torch, pandas |
| `soco/backbone.py` | Lang | `DinoFeatures` (torch.hub or HF), `freeze_all`, `unfreeze_last`, `preprocess` | torch, PIL, transformers |
| `soco/matching.py` | Lang | `sample_features`, `locate` (argmax / windowed soft-argmax), `FeatureCache`, `extract`, `alignment_score`, `predict_from_features`, `predict_pair` | backbone |
| `soco/metrics.py` | Lang | `PCKMeter` (bbox / image normalisation, `<=`) | numpy, pandas |
| `soco/evaluate.py` | Lang | `load_model`, `evaluate`, `sweep_matching`, `sweep_features` | backbone, data, matching, metrics |
| `soco/decide.py` | Lang | `recommend`: decision gate from `summary.json` + `per_category.csv` | pandas |
| `soco/finetune.py` | Lang | Stage 2: unfreeze last N blocks, soft-label matching loss, AdamW + OneCycle, AMP, `best.pth` | backbone, data, evaluate, matching |
| `soco/submit.py` | Lang | `predict_split`, `format_rows`, `check_against_sample` | data, evaluate, matching |
| `notebooks/soco_kaggle.ipynb` | Lang | **Holds the competition-specific logic**: `AMLQueryDataset`, `AMLPCKMeter` (monkey-patched into `soco.evaluate`), val split (20% of pairs, seed 42, ≤500 queries), sweeps, decide, visual check, test-CSV adapter, `predict_split` | soco |
| `outputs_first_run/**` | Lang | Results of the first run: sweeps, `stage1_best`, `submission.csv`, `test_adapted.csv` (+ 7 `.DS_Store`) | — |
| `__results___17_0.png` | Lang | Visual-check image from the notebook | — |
| `requirements.txt` | both | Same file on both branches (+ `ipykernel`, `six` added on 2026-11-03) | — |
| `environment.yml` | new (2026-11-03) | Conda env `advance`, Python 3.11, `pip -r requirements.txt` | requirements.txt |
| `.claude/*.md` | local | CLAUDE.md (task, metric, rules), pipeline.md (Lang README), firstsubmission.md (recap), instruction.md (one heading only) | — |

### 2.2 Duplicates and conflicts

| Concern | main | Lang | Conflict |
|---|---|---|---|
| **Entry point** | `python -m src.train` (+ `scripts/run_colab.py`) | `python run.py <cmd>` | Two CLIs, different flags (`--exp` vs `--config`) |
| **Config** | `src/config.py`, dict + `EXPERIMENTS`, rejects unknown keys | `soco/config.py`, dataclass | Two systems. Values disagree (table in section 5) |
| **Paths** | `src/utils/paths.py` + env vars | Absolute `/kaggle/...` strings in `Config` defaults and notebook | Hardcoded Kaggle paths |
| **Data loading** | `src/data_loader/` (empty) | `soco/data.py` (SPair + generic CSV) **and** `AMLQueryDataset` in the notebook | Three places; the one that is used is in a notebook cell |
| **Model** | `src/models/` (empty) | `soco/backbone.py`, `soco/matching.py` | — |
| **Metric** | `src/evaluate.py` docstring | `soco/metrics.py` (`<=`, bbox) **and** `AMLPCKMeter` (`<`, `scale`, min over accepted targets) | Two metrics with different boundaries; the notebook one is correct for this task |
| **Training** | `src/train.py` (run setup, W&B, checkpoints; no loop) | `soco/finetune.py` (loop; no resume, no W&B) | Must merge: main's run management + Lang's loop |
| **Evaluate** | `src/evaluate.py` (stub) | `soco/evaluate.py` | — |
| **Predict / submit** | `src/predict.py` (stub) | `soco/submit.py` + notebook adapter cell | — |
| **Diagnostics** | `src/diagnostics.py` (stub) | `soco/decide.py`, notebook visual-check cell | — |
| **Checkpoints** | `last.pt` / `best.pt` in `run_dir/checkpoints/`, atomic write, W&B | `last.pth` / `best.pth` in `out_dir/run_name/`, weights only | Names, location and content differ |
| **Seeds** | `seed=1019` | `seed=0` (config), `42` (split) | Different values |
| **Image size** | `image_size=512` (not a multiple of 14) | `img_size=518` default, **644** in the submission | Name and value differ |
| **Colab vs Kaggle runner** | `colab/run.ipynb` (thin) | `notebooks/soco_kaggle.ipynb` (thick) | Two runners; logic in notebook cells |

Bugs found while mapping (no fix without approval; see section 9):

- `soco/finetune.py` loads training data with `build_dataset(cfg, "trn")`. With `data_format="csv"`
  this reads `test.csv`, which has no targets, so Stage 2 cannot run on the competition data.
- `soco/backbone.py` (HF path) reads `hidden_states[layer]` **without the final LayerNorm**, while the
  torch.hub path uses `norm=True`. The two backbone sources return different features (section 7).
- The configs saved with the sweep results say `csv_path=test.csv, split=val`, which is not what was
  evaluated (the notebook passed `val_ds` directly).

### 2.3 Dependencies found

| Package | requirements.txt | environment.yml | Imported in code | Notebooks | Installed version (env `advance`) |
|---|---|---|---|---|---|
| python | — | 3.11 | — | — | 3.11 |
| torch, torchvision | (installed separately) | — | soco/*, src/utils | soco_kaggle | 2.15.0 (+cpu here) |
| numpy | ✓ | via pip -r | ✓ | ✓ | 2.4.6 |
| pandas | ✓ | via pip -r | ✓ | ✓ | 3.1.0 |
| pillow | ✓ | via pip -r | ✓ | ✓ | 13.0.0 |
| tqdm | ✓ | via pip -r | ✓ | — | 4.70.1 |
| transformers (≥4.56) | ✓ | via pip -r | backbone (HF path) | — | 5.22.1 |
| huggingface_hub | ✓ | via pip -r | (new: `snapshot_download`) | — | 2.4.0 |
| matplotlib | ✓ | via pip -r | — | ✓ | 3.11.2 |
| wandb | ✓ | via pip -r | src/train, checkpoint | — | 0.31.0 |
| kaggle (+ six) | ✓ | via pip -r | scripts (CLI) | — | 2.2.5 / 1.17.0 |
| ipykernel | ✓ | via pip -r | — | (Jupyter) | 7.4.0 |
| python-dotenv | — | — | **new** (`.env` loading) | — | 1.2.4 (already pulled in by a dependency) |
| google.colab | — | — | — | colab/run.ipynb | Colab only, not installable |

No notebook has a `pip install` cell.

## 3. Target structure

```text
run.py                          Single entry point: --mode train | evaluate | predict | submit
environment.yml                 Single source of dependencies (conda env `advance`)
.env.example                    Secret names, no values
.gitignore                      + .env, weights/, data/raw/, data/features/, outputs/, checkpoints
README.md                       main's README, commands updated to run.py and env `advance`

src/
  __init__.py
  config.py                     All settings (BASE + EXPERIMENTS), runtime detection, resolved paths
  metrics.py                    PCK@alpha: min distance over accepted targets / scale, strict `<`;
                                uses the official metric.py when it is present
  train.py                      Train the head (frozen mode) or head + last N blocks (finetune mode)
  evaluate.py                   Validation predictions + PCK + per-query CSV; matching/feature sweeps
  predict.py                    predict(): test predictions id,x,y; make_submission(): id,prediction
  diagnostics.py                Error breakdown, decision gate (from decide.py), visual check plot
  models/
    __init__.py                 build_model(cfg) -> CorrespondenceModel
    backbone.py                 DINOv2 through HF AutoModel; revision pin, local folder, freeze/unfreeze
    head.py                     Trainable projection head; "none" = identity (Lang behaviour)
    matching.py                 Similarity map, windowed soft-argmax, flip test, in-memory feature cache
    loss.py                     Soft-label matching loss (Lang), extended to several accepted targets
  utils/
    __init__.py
    run.py                      Run folders, seeds, git commit (main)
    checkpoint.py               Atomic save, resume, W&B publish/download (main)
    env.py                      Load secrets: .env via python-dotenv, or Colab secrets
    feature_store.py            Optional on-disk cache of frozen backbone features

data/
  README.md                     Expected data layout (CSV columns, image folders, official files)
  __init__.py
  datasets.py                   QueryDataset for train/val/test CSVs -> one sample dict format
  transforms.py                 Resize + normalise, original <-> resized coordinate mapping
  splits.py                     Build / load the validation split (by image pair)
  raw/                          (ignored) Kaggle download: train.csv, test.csv, images/, ...
  features/                     (ignored) Cached backbone features

splits/
  val_pair_ids.json             Tracked: the validation image pairs (reproduces Lang's 453 queries)

scripts/
  colab_setup.py                Clone/pull branch, install environment.yml pip list, load secrets,
                                mount Drive, export path variables
  download_weights.py           snapshot_download facebook/dinov2-* (pinned revision) to weights/
  kaggle_io.py                  Download competition data; submit a submission file

colab/
  run.ipynb                     Thin: settings cell -> scripts/colab_setup.py -> python run.py --mode ...

notebooks/
  kaggle_run.ipynb              Thin offline Kaggle runner: attach repo + weights datasets, run.py
  exploration.ipynb             (optional, later) data exploration

docs/
  PRD_reorganization.md         This document
  REORGANIZATION.md             User guide for the new repo
  EXPERIMENTS.md                Experiment log (method, config, val PCK, LB PCK), as CLAUDE.md asks
  reference/lang_first_run/     Lang's results: sweep CSVs, stage1_best summary, per_keypoint.csv,
                                submission.csv, visual_check.png (reference for acceptance checks)

tests/
  test_transforms.py            Coordinate round-trip original <-> resized
  test_splits.py                Split reproduces Lang's 453 validation query ids
  test_backbone_equivalence.py  HF vs hub features (skipped without cached weights)
  test_submission.py            Format, ids, order, NaN, bounds

weights/                        (ignored) Local DINOv2 snapshots
outputs/                        (ignored) outputs/<run_id>/{config.json, checkpoints/, predictions.csv,
                                submission.csv, metrics.json, per_query.csv}
```

**Note:** section 9, Q2 (where the dataset code lives) decides between `data/` (what you asked
for) and `src/data_loader/` (main's layout). This tree assumes `data/`.

## 4. File-by-file migration

| Source | Destination | Action | Reason |
|---|---|---|---|
| `src/config.py` | `src/config.py` | **merge** | Keep main's dict + `EXPERIMENTS` + `--set`; add all Lang settings, paths and runtime detection |
| `soco/config.py` | `src/config.py` | merge → **delete** | Settings move into `BASE`; dataclass removed |
| `src/utils/paths.py` | `src/config.py` | merge → **delete** | Paths become config values resolved per runtime (env overrides kept) |
| `src/utils/run.py` | `src/utils/run.py` | **keep** | Small edits: `prepare_run_dir` naming, seeds also set `cudnn.deterministic` |
| `src/utils/checkpoint.py` | `src/utils/checkpoint.py` | **keep** | Extended state for resume (optimizer, scheduler, scaler, epoch, step) |
| `src/utils/__init__.py` | same | keep | — |
| — | `src/utils/env.py` | **new** | `.env` / Colab secret loading in one place |
| — | `src/utils/feature_store.py` | **new** | Optional on-disk feature cache (requirement) |
| `src/data_loader/__init__.py` | `data/__init__.py` | **move** | Dataset code goes in `data/` (Q2) |
| `soco/data.py` | `data/datasets.py` | **rewrite** | Keep only the CSV query reader; drop `SPairDataset` (not used for the competition, see Q6) |
| notebook `AMLQueryDataset` | `data/datasets.py` | **move** | The working competition reader, out of the notebook |
| notebook split cell | `data/splits.py` + `splits/val_pair_ids.json` | **move** | Same algorithm (20% of pairs, seed 42), saved once and tracked |
| notebook test adapter cell | `data/datasets.py` | **delete** (absorbed) | `QueryDataset` reads `test.csv` directly; no `test_adapted.csv` |
| `soco/backbone.py` `preprocess` | `data/transforms.py` | **move** | Transforms belong to data; same resize + ImageNet normalisation |
| `soco/backbone.py` `DinoFeatures` | `src/models/backbone.py` | **rewrite** | HF `AutoModel` only, LayerNorm applied (= hub `norm=True`), revision pin, local folder |
| `src/models/__init__.py` | `src/models/__init__.py` | **rewrite** | `build_model(cfg)` |
| `soco/matching.py` | `src/models/matching.py` | **move** | Logic unchanged; settings read from config |
| — | `src/models/head.py` | **new** | Trainable head; identity by default so the Lang result is unchanged |
| `soco/finetune.py` `matching_loss` | `src/models/loss.py` | **move** | Same loss; several accepted targets → mixture label (Q4) |
| `soco/finetune.py` loop | `src/train.py` | **merge** | Lang's loop + main's run dir, resume, W&B, checkpoints |
| `src/train.py` | `src/train.py` | **merge** | Keeps CLI setup; loop added; called from `run.py` |
| `soco/metrics.py` + notebook `AMLPCKMeter` | `src/metrics.py` | **merge** | One metric: min over accepted targets / `scale`, `<` (notebook). Bbox/SPair paths removed |
| `soco/evaluate.py` | `src/evaluate.py` | **merge** | `evaluate`, `sweep_matching`, `sweep_features`; no monkey-patching |
| `src/evaluate.py` | `src/evaluate.py` | merge | Stub replaced |
| `soco/submit.py` | `src/predict.py` | **merge** | `predict()` → `predictions.csv` (id,x,y); `make_submission()` → `submission.csv` |
| `src/predict.py` | `src/predict.py` | merge | Stub replaced |
| `soco/decide.py` | `src/diagnostics.py` | **move** | Decision gate is a diagnostic |
| notebook visual-check cell | `src/diagnostics.py` | **move** | `plot_predictions()` |
| `src/diagnostics.py` | `src/diagnostics.py` | merge | Stub replaced |
| `run.py` (Lang) | `run.py` | **rewrite** | `--mode train/evaluate/predict/submit` + main's `--exp/--set/--resume/--dry-run` |
| `soco/__init__.py` | — | **delete** | Package emptied |
| `scripts/setup_colab.py` | `scripts/colab_setup.py` + `scripts/kaggle_io.py` | **split + rename** | Install from `environment.yml`; Kaggle download moves to `kaggle_io.py` |
| `scripts/run_colab.py` | `run.py` (preflight) | **merge → delete** | GPU / secret checks become `run.py` preflight; the notebook calls `run.py` |
| — | `scripts/download_weights.py` | **new** | `snapshot_download` to `weights/` |
| `colab/run.ipynb` | `colab/run.ipynb` | **rewrite** | Thin: settings → `colab_setup.py` → `run.py --mode` |
| `notebooks/soco_kaggle.ipynb` | `notebooks/kaggle_run.ipynb` | **rewrite** | Thin offline Kaggle runner; its logic now lives in `data/` and `src/` |
| `outputs_first_run/**` (minus `.DS_Store`) | `docs/reference/lang_first_run/` | **move** | Reference for the acceptance checks; small files (Q7) |
| `outputs_first_run/**/.DS_Store` | — | **delete** | macOS junk |
| `outputs_first_run/test_adapted.csv` | — | **delete** | Derived from `test.csv`; no longer needed |
| `__results___17_0.png` | `docs/reference/lang_first_run/visual_check.png` | **move** | — |
| `requirements.txt` | `environment.yml` | merge → **delete** | Single dependency file (requirement) |
| `environment.yml` | `environment.yml` | **rewrite** | Pinned packages, torch included, `pip:` section; adds `python-dotenv` and `pytest` |
| — | `tests/*.py` | **new** | Acceptance checks as tests |
| — | `.env.example` | **new** | Secret names |
| `.gitignore` | `.gitignore` | **edit** | Add `.env`, `weights/`, `data/raw/`, `data/features/`, `checkpoints/`, `.DS_Store`; keep `data/*.py`, `data/README.md` tracked |
| `README.md` | `README.md` | **edit** | New commands; points to `docs/REORGANIZATION.md` |
| — | `data/README.md`, `docs/REORGANIZATION.md`, `docs/EXPERIMENTS.md` | **new** | Documentation |
| `.claude/*` | — | **untouched** | Local, gitignored |

## 5. Config design (`src/config.py`)

Main's style is kept: one `BASE` dict, named `EXPERIMENTS` that list only the differences, `--set
key=value` overrides, and unknown keys rejected. **Rule:** when the branches disagree, the Lang value
wins (the submission works with it). The **Values used** column is the configuration that produced
the Lang submission.

### 5.1 Runtime and paths

| Key | Value | Origin | Notes |
|---|---|---|---|
| `runtime` | `"auto"` → `local` / `colab` / `kaggle` | new | Kaggle: `KAGGLE_KERNEL_RUN_TYPE` env; Colab: `COLAB_RELEASE_TAG` env or `google.colab` importable |
| `competition` | `"aml-challenge-2026-27"` | main (`COMPETITION`) | |
| `data_dir` | local `data/raw`, Colab `/content/data/raw`, Kaggle `/kaggle/input/competitions/aml-challenge-2026-27` | main (`RAW_DATA_DIR`) + Lang (Kaggle path) | Env `SC_DATA_DIR` still overrides |
| `output_dir` | local `outputs`, Colab `/content/drive/MyDrive/aml_semantic_correspondence/outputs`, Kaggle `/kaggle/working/outputs` | main (`OUTPUT_DIR`, Drive path) + Lang | Env `SC_OUTPUT_DIR` still overrides |
| `weights_dir` | local `weights`, Colab `/content/drive/MyDrive/aml_semantic_correspondence/weights`, Kaggle `/kaggle/input/dinov2-base-hf` | new | Kaggle dataset name to be confirmed (Q5) |
| `features_dir` | `data/features` (Colab: on Drive) | new | On-disk feature cache |
| `splits_dir` | `splits` | main (`SPLITS_DIR`) | |
| `train_csv`, `test_csv`, `sample_submission_csv` | `train.csv`, `test.csv`, `sample_submission.csv` (relative to `data_dir`) | Lang (notebook) | |
| `image_root` | `data_dir` (CSV image paths are relative to it, e.g. `images/test/img_0554.jpeg`) | Lang (notebook) | |

### 5.2 Reproducibility, logging, run management

| Key | Value | Origin | Notes |
|---|---|---|---|
| `seed` | **0** | Lang (`seed=0`); main had 1019 | Lang wins |
| `split_seed` | **42** | Lang (notebook) | |
| `val_fraction` | 0.2 (of unique `pair_id`s) | Lang (notebook) | |
| `max_val_queries` | 500 | Lang (notebook) | Gave 453 queries |
| `deterministic` | True | new (CLAUDE.md) | `cudnn.deterministic`, no `benchmark` |
| `num_workers` | 2 | Lang | |
| `device` | `"auto"` | Lang | |
| `wandb_project` | `sapienza-aml-semantic-correspondence` | main | |
| `wandb_entity` | `pfingsten-sapienza-universit-di-roma` | main | |
| `use_wandb` | True for train, False for evaluate/predict | new | Lang had no W&B |
| `publish_best` | True | main | |
| `init_artifact` | None | main | |

### 5.3 Model

| Key | Value | Origin | Notes |
|---|---|---|---|
| `model_id` | `facebook/dinov2-base` | Lang `backbone=dinov2_vitb14` → HF ID; main had `model="dino_baseline"` | Replaces `backbone`, `backbone_source`, `hub_repo`, `hub_local`, `weights_path`, `model` |
| `model_revision` | commit hash of `facebook/dinov2-base` (filled in at implementation) | new | Pins the weights |
| `local_weights` | True when `weights_dir/<model_id>` exists | new | Offline Kaggle |
| `feature_layer` | **-1** | Lang (`layer`, best in sweep) | |
| `image_size` | **644** | Lang (`img_size`; 518 default, 644 in submission); main had 512 | Name from main, value from Lang; must be a multiple of 14 |
| `patch_size` | 14 (read from the model, checked) | Lang | |
| `image_mean`, `image_std` | ImageNet (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) | Lang | Same as HF processor |
| `resize_mode` | `"square"`, bicubic | Lang | Not changed (Q8) |
| `head` | **`"none"`** (identity) | new | `"linear"` / `"mlp"` for training; `"none"` reproduces Lang |
| `head_dim` | 768 | new | Output dim of the head |

### 5.4 Matching (inference)

| Key | Value | Origin |
|---|---|---|
| `matcher` | `"soft_window"` | Lang |
| `soft_window` | 5 | Lang |
| `soft_temperature` | 0.04 | Lang |
| `feat_upsample` | 2 | Lang |
| `flip_test` | False | Lang |
| `feature_cache_size` | 256 images (in memory) | Lang (`FeatureCache(max_items=256)`) |
| `cache_features_to_disk` | False | new (optional requirement) |

### 5.5 Training (head in frozen mode, Phase 2 in finetune mode)

| Key | Value | Origin | Notes |
|---|---|---|---|
| `train_mode` | `"frozen"` | new | `"frozen"` \| `"finetune"` |
| `unfreeze_blocks` | 2 | Lang (`ft_unfreeze_blocks`) | Only in finetune mode |
| `batch_size` | **4** | Lang (`ft_batch_size`); main had 8 | |
| `epochs` | **2** | Lang (`ft_epochs`); main had 10 | |
| `lr_head` | 1e-4 | main (`lr`) | Lang had no head |
| `lr_backbone` | **1e-5** | Lang (`ft_lr`) | |
| `weight_decay` | 0.05 | Lang (hardcoded in finetune.py) | |
| `warmup_pct` | 0.05 | Lang (OneCycle `pct_start`, hardcoded) | |
| `grad_clip` | 1.0 | Lang (hardcoded) | |
| `grad_accum_steps` | 1 | new | |
| `amp` | True (fp16, CUDA only) | Lang | |
| `loss_temperature` | 0.03 | Lang (`ft_temperature`) | |
| `label_sigma` | 1.0 (patches) | Lang (`ft_label_sigma`) | |
| `flip_augment` | True | Lang (`ft_pair_flip_aug`) | |
| `max_train_queries` | 0 (all) | Lang (`ft_max_train_pairs`) | |
| `checkpoint_every_steps` | 0 (epoch only) | new | Colab disconnect safety |

### 5.6 Evaluation and submission

| Key | Value | Origin | Notes |
|---|---|---|---|
| `pck_alpha` | 0.10 | CLAUDE.md / Lang notebook | Main score |
| `pck_alphas_report` | (0.05, 0.10, 0.15) | Lang | Extra columns in reports |
| `pck_strict` | True (`d/L < alpha`) | Lang notebook | Replaced by official `metric.py` if present (Q3) |
| `submission_columns` | `["id", "prediction"]` | Lang / CLAUDE.md | |
| `prediction_decimals` | 2 | Lang (`fmt_xy`) | `"[75.81,228.38]"` |
| `clip_to_image` | True | CLAUDE.md | Lang did not clip; values are unchanged when already in bounds |
| `decide_target` | 70.0 | Lang (`decide`) | |
| `decide_sanity_floor` | 35.0 | Lang | |

**Not applicable:** "number of classes" and "label mapping". The task is point correspondence, not
classification, and neither branch has classes. The CSVs have no category column (the Lang run
used `category="all"`). Q9 asks whether a category column exists.

**Removed Lang settings:** `data_root`, `data_format`, `split`, `csv_path`, `image_dir`, `categories`,
`max_pairs`, `pck_norm` (SPair-only or replaced by the paths above), and `out_dir`, `run_name`
(replaced by main's run folders). `ft_checkpoint` is replaced by `--run RUN_ID` / `--checkpoint`.

## 6. Naming conventions

| Thing | Name used everywhere | Replaces |
|---|---|---|
| HF model | `model_id`, `model_revision` | `backbone`, `model`, `hub_repo`, `weights_path` |
| Input size | `image_size` | `img_size` |
| Query id (`id` column) | `query_id` | `pair_id` (Lang reused it for queries) |
| Image pair id | `pair_id` | `image_pair_id` |
| Image paths | `src_image`, `trg_image` | `source_image`, `src_path`, `trg_path` |
| Query point | `src_xy` (`[2]`, original pixels) | `src_kps`, `source_x/y` |
| Accepted targets | `trg_xys` (`[N, 2]`) | `accepted_targets`, `target_points`, `trg_kps` |
| Object size L | `scale` | — |
| Prediction | `pred_xy` (`[2]`, original pixels) | `pred` |
| Feature maps | `src_feats`, `trg_feats` (`[C, h, w]`) | `fs`, `ft` |
| Config object | `cfg` (dict) | `Config` dataclass |
| Run folder | `run_id`, `run_dir` | `run_name`, `out_dir` |
| Checkpoints | `run_dir/checkpoints/{last,best}.pt` | `last.pth`, `best.pth` |
| Prefixes | `src_` / `trg_` for source / target | `source_` / `target_` |
| Functions | verb first: `build_model`, `load_split`, `predict_point`, `write_submission` | — |
| Modes | `train`, `evaluate`, `predict`, `submit` | `eval`, `finetune`, `sweep`, ... |

Coordinates are always `(x, y)`, origin top-left, in **original image pixels** outside
`data/transforms.py`. Resized coordinates appear only inside the model code and are named `*_resized`.

## 7. DINOv2 mapping and equivalence check

**What Lang used:** `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`. On Kaggle this was
loaded offline from a local copy of the repo and `dinov2_vitb14.pth`.

| Setting | Value |
|---|---|
| Model | ViT-B/14, 86M parameters, 768-dim, 12 blocks, no registers |
| Input | Square bicubic resize (PIL) to 644×644, /255, ImageNet mean/std |
| Features | `get_intermediate_layers(x, n=[11], reshape=True, norm=True)`: patch tokens of the last block, **after the final LayerNorm**, CLS dropped, L2-normalised per patch |
| Grid | 644 / 14 = 46×46 patches, then upsampled 2× to 92×92 for matching |

**Hugging Face equivalent:** `facebook/dinov2-base`, loaded with `AutoModel`.

| Lang / torch.hub | HF ID |
|---|---|
| `dinov2_vits14` | `facebook/dinov2-small` |
| **`dinov2_vitb14` (used)** | **`facebook/dinov2-base`** |
| `dinov2_vitl14` | `facebook/dinov2-large` |
| `dinov2_vitg14` | `facebook/dinov2-giant` |
| `dinov2_vit*14_reg` | `facebook/dinov2-with-registers-{small,base,large,giant}` |

**How the HF version must be used to match:**

1. **Final LayerNorm.** Take `outputs.hidden_states[-1]` (or `[layer]`) and apply `model.layernorm`.
   This equals hub `norm=True`. Lang's own HF path skipped it.
   `outputs.last_hidden_state` already has it for layer -1.
2. **Token pooling.** Patch tokens only: drop index 0 (CLS), plus 4 register tokens for `_reg`
   models. No CLS, no mean pooling. Matching needs the per-patch map.
3. **Layer indexing.** HF `hidden_states` has 13 entries (embeddings + 12 blocks).
   `hidden_states[-1]` = hub block 11, `[-2]` = hub block 10. Same as Lang's `layer % 12`.
4. **Preprocessing.** **Do not use `AutoImageProcessor` defaults.** They resize the shortest edge
   to 256 and **center-crop to 224**. That cuts off part of the image and breaks the coordinate
   mapping. Use `AutoImageProcessor` only with `size={"height": S, "width": S},
   do_center_crop=False` (bicubic, ImageNet stats), or keep Lang's `preprocess`. The plan uses
   the processor's mean/std/resample values, read from the pinned revision, with an explicit
   square resize, so the coordinate mapping stays exact.
5. **Position embeddings.** Both interpolate the 37×37 pretrained grid to 46×46 with bicubic
   interpolation. To be confirmed by the numbers below.

**Measured equivalence** (synthetic 500×375 image, ViT-B/14, cosine similarity between hub and HF
patch features, 1.0 = identical):

| Input size | Layer | HF, no LayerNorm (Lang's HF path) | HF + LayerNorm, Lang preprocessing | HF + LayerNorm, HF processor (square, no crop) |
|---|---|---|---|---|
| 518 (37×37, native grid) | -1 | 0.756 (min 0.679) | **1.00000** (min 1.00000) | 1.00000 (min 0.99995) |
| 518 | -2 | 0.808 (min 0.614) | **1.00000** (min 1.00000) | 1.00000 (min 0.99995) |
| **644 (46×46, used)** | -1 | 0.753 (min 0.670) | **0.99969** (min 0.99203) | 0.99968 (min 0.99214) |
| 644 | -2 | 0.801 (min 0.615) | **0.99971** (min 0.99272) | 0.99971 (min 0.99284) |

The default `AutoImageProcessor` output is `(1, 3, 224, 224)`, which confirms the crop.

What this shows:
- **Weights are identical.** At 518 px no position-embedding interpolation happens, and HF + LayerNorm
  matches hub exactly.
- **At 644 px there is a small difference** (mean cosine 0.9997, worst patch 0.992). It comes from
  the position-embedding interpolation: hub and HF resize the 37×37 grid to 46×46 in slightly
  different ways. Argmax locations can move by one cell on a few queries, so the PCK can change by
  a few tenths. The ±0.5 acceptance below covers it. If it does not, the hub interpolation can be
  copied into `backbone.py` (a few lines) so HF reproduces it exactly.
- **Without the LayerNorm the features are different** (cosine ≈ 0.75). Lang's HF option would not
  have reproduced the hub results. The new code always applies the LayerNorm.
- The HF processor with an explicit square size gives almost the same input as Lang's PIL resize
  (max pixel difference 0.035 after normalisation, from rounding). Either is fine.

Script: run once from the scratchpad (not part of the repo). The same check becomes
`tests/test_backbone_equivalence.py`, skipped when the weights are not cached.

**Acceptance for the swap:** on the 453 validation queries, the HF pipeline must give PCK@0.10
within ±0.5 points of 40.4. On the test set, ≥ 99% of predictions must lie within 1 px of the Lang
`submission.csv`.

## 8. Phase 2 plan: fine-tuning in Colab (prepared, not run)

**Switch:** `train_mode = "frozen" | "finetune"` (set with `--set train_mode=finetune` or the
`finetune` experiment).

| | frozen (Phase 1) | finetune (Phase 2) |
|---|---|---|
| Backbone params | `requires_grad=False`, always `eval()` | Last `unfreeze_blocks` (2) blocks + final LayerNorm trainable; the rest frozen and in `eval()` |
| Trained | Head only | Head + unfrozen blocks |
| Optimiser | AdamW, `lr_head` 1e-4 | AdamW with two parameter groups: head `lr_head` 1e-4, backbone `lr_backbone` 1e-5; wd 0.05 |
| Schedule | OneCycle, warmup 5% | Same |
| Precision | AMP fp16 on CUDA | AMP fp16 + GradScaler (T4 has no bf16); `grad_accum_steps` optional |
| Feature cache | On-disk cache allowed (backbone fixed) | **Disabled** (features change every step) |
| Validation | Every epoch on `splits/val_pair_ids.json`, PCK@0.10 | Same |

**Loss:** Lang's soft-label matching loss. For every training query, the softmax over target patches
(temperature 0.03) is trained against a Gaussian label (σ = 1 patch) at the target. Several accepted
targets give a **normalised mixture of Gaussians**, so any accepted wheel counts. This extends
Lang's single-target loss (Q4).

**Training data:** `train.csv` minus the validation pairs. Random horizontal flip of both images
(Lang). This fixes the Lang bug where finetune read `test.csv`.

**Checkpoints and resume** (main's design, extended):
- `output_dir` on Colab is on Google Drive, so `run_dir/checkpoints/last.pt` survives a disconnect.
- `last.pt` holds: trainable weights (head + unfrozen blocks), optimiser, scheduler, scaler,
  epoch, step, RNG states, config, `git_commit`. It is written atomically every epoch and,
  optionally, every `checkpoint_every_steps`.
- `best.pt` is written when val PCK improves and uploaded to W&B at the end (`publish_best`).
- Resume: `python run.py --mode train --resume <run_id>`. In the notebook, set `RESUME="<run_id>"`.
  The W&B run resumes with the same id.

**Colab flow:** GPU runtime → `colab/run.ipynb` → `scripts/colab_setup.py` (pull branch, install,
secrets, Drive, data, weights) → `run.py --mode train --exp finetune` → `--mode evaluate --run <id>`
→ `--mode submit --run <id>`.

**Budget estimate (to check on Colab):** ViT-B/14 at 644 px, batch 4, fp16 on a T4. Memory should be
comfortable. Time per epoch depends on the training-set size, unknown until the data is downloaded.

## 9. Risks and open questions

**Questions for you:**

| # | Question | My recommendation |
|---|---|---|
| Q1 | **Branch `v1` does not exist.** The merge of Lang into main is still uncommitted on `main` (staged, `MERGE_HEAD` = `0616d1f`). Should I commit the merge on a new local branch `v1` and branch `refactor/v1-reorg` from it? | Yes: `git switch -c v1`, commit the merge there (`main` untouched), then branch from it |
| Q2 | Your spec puts dataset code in **`data/`**, but main's layout has **`src/data_loader/`**, and `data/` is where main downloads the raw data. Which one? | `data/` as you asked, with the raw data in the ignored `data/raw/`. Alternative: `src/data_loader/` + `data/README.md` |
| Q3 | **`metric.py`**: CLAUDE.md says to use it, but main's `evaluate.py` says it is not in the Kaggle download. Do you have it? | If it exists, `src/metrics.py` calls it; otherwise the notebook metric (`<`, `scale`) is the fallback, as now |
| Q4 | **Trainable head.** Your brief says "frozen DINOv2 with a trainable head", but neither branch has a head: Lang's result is training-free matching. OK to add a head that is **identity by default** (so the submission is unchanged) and a residual linear/MLP head trained with Lang's matching loss? And OK to extend the loss to several accepted targets? | Yes to both. `head="none"` reproduces Lang exactly |
| Q5 | **Local data:** there is no competition data and no Kaggle credentials on this machine, so the local end-to-end check (train/evaluate/predict/submit) cannot run yet. Can you put `KAGGLE_USERNAME`/`KAGGLE_KEY` in `.env`, or the data in `data/raw/`? Also: the name of the Kaggle dataset for the HF weights | Put the keys in `.env`; I will download with `scripts/kaggle_io.py` |
| Q6 | Drop SPair-71k support (`SPairDataset`, bbox normalisation)? It is not used for the competition | Drop it; add back only if you want SPair pre-training |
| Q7 | Keep Lang's first-run results in git as `docs/reference/lang_first_run/` (≈ 200 KB)? main's rule is "outputs not in git" | Yes, as a reference for acceptance; delete later |
| Q8 | The square resize stretches images. Keep it (Lang behaviour) for now? | Keep it; test aspect-preserving resize later as a logged experiment |
| Q9 | Does `train.csv` have a category column? Then per-category reports come back | Unknown until the data is downloaded |
| Q10 | Pin exact versions in `environment.yml` (the ones tested in `advance`: torch 2.15.0, transformers 5.22.1, …)? | Yes, pin `==` for reproducibility |

**Risks:**

| Risk | Effect | Mitigation |
|---|---|---|
| HF and hub features differ slightly | PCK can move by a few tenths | Equivalence check (section 7) + ±0.5 PCK acceptance on val |
| HF processor defaults (crop 224) | Wrong coordinates, large drop | Explicit square resize, unit test on coordinate round-trip |
| Validation split not reproduced exactly | Val numbers not comparable with 40.4 | `data/splits.py` must give exactly the 453 ids in `per_keypoint.csv`; checked in tests |
| Library versions differ from Kaggle's (pandas 3, numpy 2.4) | Small numeric differences | Pinned env; compare against the reference submission |
| Kaggle offline: `python-dotenv` / `wandb` unavailable | Import errors | Optional imports; W&B off by default outside `train` |
| `.claude/` is gitignored (your local change) | CLAUDE.md conventions not shared with teammates | Your decision; no action planned |
| fp16 overflow in fine-tuning | NaN loss | GradScaler + grad clipping (Lang), NaN check aborts the step |

## 10. Acceptance criteria

- [ ] All `.claude/*.md` files read (CLAUDE.md, pipeline.md, firstsubmission.md, instruction.md), and the work follows CLAUDE.md (validation first, read-only official files, experiment log, deterministic inference, submission checks).
- [ ] The repo follows main's structure (section 3); `soco/` is gone; each function exists once (checked with `grep`).
- [ ] `src/config.py` holds every setting. No hardcoded paths or numbers elsewhere: a `grep` for `/kaggle`, `/content`, and numeric literals in `src/` and `data/` is reviewed.
- [ ] Dataset code is in `data/` (or `src/data_loader/`, per Q2); `data/README.md` describes the layout; no raw data committed.
- [ ] `python run.py --mode train|evaluate|predict|submit` runs locally (CPU, small `--set max_train_queries=…` for train).
- [ ] `colab/run.ipynb` only sets options and calls `scripts/colab_setup.py` and `run.py`.
- [ ] DINOv2 is loaded from `facebook/dinov2-base` at a pinned revision, also from a local folder; features match the hub model (section 7).
- [ ] `splits/val_pair_ids.json` reproduces the 453 Lang validation queries exactly.
- [ ] `run.py --mode evaluate` (head none, Lang config) gives val PCK@0.10 = 40.4 ± 0.5.
- [ ] `run.py --mode submit` gives a file with the same columns, ids and order as `sample_submission.csv` and the Lang file; no NaN; points inside the image; ≥ 99% of points within 1 px of the Lang `submission.csv`.
- [ ] `train_mode=finetune` builds the right parameter groups, uses AMP, saves `last.pt` and resumes from it (dry run on a few queries; not run in full).
- [ ] `requirements.txt` deleted; `conda env create -f environment.yml` builds a working env (tested by recreating `advance`).
- [ ] `.env.example` exists; `.env`, `weights/`, `data/raw/`, `data/features/`, `outputs/` are ignored; `git log -p` contains no secrets.
- [ ] `docs/PRD_reorganization.md`, `docs/REORGANIZATION.md`, `docs/EXPERIMENTS.md` written and accurate.
- [ ] Work is on `refactor/v1-reorg` in small commits; nothing pushed until you ask; `main` untouched.
