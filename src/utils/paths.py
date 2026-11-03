"""Central project paths. Change locations with env vars, not in the code."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SC_DATA_DIR", REPO_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = REPO_ROOT / "splits"

# On Colab the notebook points this at Google Drive, so runs survive a disconnect.
OUTPUT_DIR = Path(os.environ.get("SC_OUTPUT_DIR", REPO_ROOT / "outputs"))
