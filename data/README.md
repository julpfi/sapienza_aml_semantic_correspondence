# Data

This folder holds the **dataset code** (tracked) and the **raw data** (ignored by git).

```text
data/
  README.md        this file
  __init__.py
  datasets.py      QueryDataset: one CSV row = one query -> one sample dict
  transforms.py    resize + normalise, original <-> resized coordinates, flips
  splits.py        validation split by image pair (saved to splits/val_pair_ids.json)
  raw/             (ignored) the Kaggle download, see below
  features/        (ignored) optional cache of frozen backbone features
```

Never commit anything under `raw/` or `features/`.

## Getting the data

```bash
python scripts/kaggle_io.py download      # needs KAGGLE_USERNAME / KAGGLE_KEY in .env
```

This downloads and unzips the competition into `data_dir`:

| Runtime | `data_dir` |
|---|---|
| local | `data/raw/` |
| Colab | `/content/data/raw/` (re-downloaded each session by `scripts/colab_setup.py`) |
| Kaggle | `/kaggle/input/competitions/aml-challenge-2026-27/` (attached input) |

Set `SC_DATA_DIR` to use another folder.

## Expected layout of `data_dir`

```text
train.csv                 training queries, with accepted targets
test.csv                  test queries, without targets
sample_submission.csv     id,prediction: every test id once, in the expected order
make_submission.py        official converter id,x,y -> id,prediction (read-only)
metric.py                 official metric, if provided (read-only; used by evaluate when present)
images/...                image files; CSV paths are relative to data_dir (e.g. images/test/img_0554.jpeg)
```

Columns (as used by the Lang notebook; the mapping is `csv_columns` in `src/config.py`):

| Column | In | Meaning | Name in the code |
|---|---|---|---|
| `id` | train, test | query id, one submission row | `query_id` |
| `pair_id` | train (test: optional) | image pair id; the validation split holds out whole pairs | `pair_id` |
| `source_image`, `target_image` | train, test | image paths relative to `data_dir` | `src_image`, `trg_image` |
| `source_x`, `source_y` | train, test | query point in original source pixels | `src_xy` |
| `target_points` | train | JSON list of **all accepted** targets `[[x, y], ...]` | `trg_xys` |
| `scale` | train | object size L for PCK (longest side of the target bounding box) | `scale` |
| `category` | if present | object category; otherwise everything is `"all"` | `category` |

Coordinates are `(x, y)` in original pixels, origin top-left, x to the right, y down.

**To confirm on the real data** (not checked here because the data was not available while
reorganising): whether `test.csv` has `pair_id`, whether a `category` column exists, and whether
`metric.py` is in the download. If a column name differs, change `csv_columns`; the code does
not need to change.

## Validation split

`splits/val_pair_ids.json` lists the held-out image pairs: 20% of the unique `pair_id`s, shuffled
with numpy seed 42, which is the Lang notebook algorithm. Validation uses at most 500 of those
queries (453 in practice). The first `evaluate` or `train` run creates the file; commit it so
everyone validates on the same pairs. It gives exactly Lang's 453 validation queries.
