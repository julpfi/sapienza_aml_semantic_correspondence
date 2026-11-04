# Reorganization guide (v1 → refactor/v1-reorg)

The `v1` branch merged main (project structure) and Lang (the working Kaggle pipeline) without
integrating them. This branch turns them into one codebase. It uses main's layout, Lang's logic
and results, and is ready for fine-tuning on Colab (Phase 2).
Plan and decisions: [PRD_reorganization.md](PRD_reorganization.md). Commands: [../RUN.md](../RUN.md).

## Final structure

```text
run.py                      single entry point: --mode train | evaluate | submit
environment.yml             the only dependency file (conda env `advance`)
.env.example                secret names; copy to .env (ignored)
RUN.md                      how to run everything (local, Colab, Kaggle)

src/
  config.py                 the only configuration (BASE), runtime detection and paths
  train.py                  frozen-head training / Phase 2 fine-tuning, resume, W&B
  evaluate.py               validation PCK
  predict.py                test predictions and the submission file
  diagnostics.py            error breakdown, visual check
  metrics.py                PCK@alpha (nearest accepted target / scale, strict <); official metric.py if present
  models/
    __init__.py             CorrespondenceModel + build_model
    backbone.py             DINOv2 from facebook/ on Hugging Face (pinned revision, local folder)
    head.py                 trainable head; "none" = identity (Lang behaviour)
    matching.py             similarity, windowed soft-argmax, flip test, in-memory cache
    loss.py                 soft-label matching loss with all accepted targets
  utils/
    run.py                  run folders, saved configs, seeds, git commit
    checkpoint.py           atomic save, resume, W&B publish/download
    env.py                  secrets from .env or Colab Secrets
    feature_store.py        optional on-disk cache of frozen features

data/
  README.md                 expected data layout and columns
  datasets.py               QueryDataset (train/val/test from the competition CSVs)
  transforms.py             resize/normalise and coordinate mapping
  splits.py                 validation split by image pair (Lang algorithm)
  raw/, features/           (ignored) downloaded data, feature cache

splits/val_pair_ids.json    tracked validation pairs (written by the first run on the real data)
scripts/
  colab_setup.py            Colab: pull, install, secrets, Drive, data, weights
  download_weights.py       snapshot_download of the DINOv2 weights to weights/
  kaggle_io.py              download the competition data, submit a file
colab/run.ipynb             thin Colab runner
notebooks/kaggle_run.ipynb  thin offline Kaggle runner
docs/
  PRD_reorganization.md     plan and decisions
  REORGANIZATION.md         this guide
  decisions.md              decision log
  run_explanation.md        why there are several "run" files
  reference/lang_first_run/ Lang's first-run results (reference for checks)
```

## What moved where

| Before (branch) | Now | Notes |
|---|---|---|
| `soco/config.py` (Lang), `src/config.py` (main), `src/utils/paths.py` (main) | `src/config.py` | One `BASE` dict; Lang values win; paths per runtime |
| `soco/data.py` (Lang), notebook `AMLQueryDataset` (Lang), `src/data_loader/` (main, empty) | `data/datasets.py` | SPair-71k adapter dropped (not used for the competition) |
| notebook validation-split cell (Lang) | `data/splits.py` + `splits/val_pair_ids.json` | Same algorithm, saved and tracked |
| notebook test adapter cell, `test_adapted.csv` (Lang) | (not needed) | `QueryDataset` reads `test.csv` directly |
| `soco/backbone.py` `preprocess` (Lang) | `data/transforms.py` | Unchanged |
| `soco/backbone.py` `DinoFeatures` (Lang, torch.hub) | `src/models/backbone.py` | Hugging Face `AutoModel`, identical features (below) |
| `soco/matching.py` (Lang) | `src/models/matching.py` + `src/models/__init__.py` | Logic unchanged |
| `soco/finetune.py` loss (Lang) | `src/models/loss.py` | Several accepted targets supported |
| `soco/finetune.py` loop (Lang), `src/train.py` (main, no loop) | `src/train.py` | Lang loop + main's run folders, resume, W&B |
| `soco/metrics.py` + notebook `AMLPCKMeter` (Lang) | `src/metrics.py` | No more monkey-patching |
| `soco/evaluate.py` (Lang), `src/evaluate.py` (main, stub) | `src/evaluate.py` | |
| `soco/submit.py` (Lang), `src/predict.py` (main, stub) | `src/predict.py` | Same output format |
| `soco/decide.py`, notebook visual check (Lang), `src/diagnostics.py` (main, stub) | `src/diagnostics.py` | Decision gate removed (D-005) |
| `run.py` (Lang), `python -m src.train` (main) | `run.py --mode ...` | One CLI |
| `scripts/setup_colab.py`, `scripts/run_colab.py` (main) | `scripts/colab_setup.py`, `scripts/kaggle_io.py`, `run.py` checks | |
| `notebooks/soco_kaggle.ipynb` (Lang) | `notebooks/kaggle_run.ipynb` | Thin; logic moved to code |
| `colab/run.ipynb` (main) | `colab/run.ipynb` | Calls `colab_setup.py` and `run.py` |
| `requirements.txt` (both) | `environment.yml` | Deleted; all packages pinned in the env file |
| `outputs_first_run/`, `__results___17_0.png` (Lang) | `docs/reference/lang_first_run/` | `.DS_Store` and `test_adapted.csv` removed |

Renamed settings (old → new): `backbone`/`model`/`hub_repo`/`weights_path` → `model_id` (+ `model_revision`),
`img_size` → `image_size`, `layer` → `feature_layer`, `ft_lr` → `lr_backbone`, `lr` → `lr_head`,
`ft_epochs`/`ft_batch_size` → `epochs`/`batch_size`, `ft_unfreeze_blocks` → `unfreeze_blocks`,
`ft_temperature` → `loss_temperature`, `ft_label_sigma` → `label_sigma`, `ft_pair_flip_aug` → `flip_augment`,
`out_dir`/`run_name` → `output_dir`/`run_id`. Sample keys: `src_kps` → `src_xy`,
`trg_kps`/`accepted_targets` → `trg_xys`, query `pair_id` → `query_id`. Checkpoints:
`last.pth`/`best.pth` → `checkpoints/last.pt`/`best.pt`.

## Equivalence with the Lang pipeline

Checked on 2026-11-03 (scripts run against Lang's code exported from `v1`):

- **Features:** `facebook/dinov2-base` (revision `f9e44c8`) vs torch.hub `dinov2_vitb14` at 644 px:
  cosine 1.000000 (min 0.999999), max abs difference 2.4e-7. This needs the final LayerNorm and
  torch.hub's position-embedding interpolation (`pos_embed_interp="hub"`). The transformers default
  interpolation would give min cosine 0.885 on a real photo.
- **Predictions:** on a mock test set built from real photos, Lang's `soco` code (as in its
  notebook) and `run.py --mode submit` give the **same submission file**: identical ids,
  order and prediction strings, max difference 0.0 px.
- **Validation split:** same algorithm and seeds as Lang's notebook. On the real data it gives the 453 ids
  in `docs/reference/lang_first_run/stage1_best/per_keypoint.csv` (checked before the test suite was removed, D-004).

## Environment

```bash
conda env create -f environment.yml && conda activate advance   # local
```

Colab does not use conda: `scripts/colab_setup.py` pip-installs the `pip:` list of
`environment.yml` and keeps Colab's CUDA torch. Details and every command: [../RUN.md](../RUN.md).

## Behaviour changes to know about

- **Validation metric:** strict `d/L < 0.10` with the nearest accepted target (the Lang notebook
  metric). Lang's `soco/metrics.py` used `<=` and SPair bounding boxes; that path is gone.
- **Clipping:** predictions are clipped to the image. Lang's soft-argmax never leaves the image,
  so its results do not change.
- **Fine-tuning:** Lang's version read `test.csv` as training data (it had no targets); it now
  trains on `train.csv` minus the validation pairs, and its loss accepts all accepted targets.
- **Lang's Hugging Face option** skipped the final LayerNorm (cosine ≈ 0.75 to the hub features). The
  new backbone always applies it.
