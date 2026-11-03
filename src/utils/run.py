"""Helpers for creating reproducible run folders."""

import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.config import BASE
from src.utils.paths import OUTPUT_DIR, REPO_ROOT


def git_commit():
    """Return the current commit, marked -dirty if there are uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def load_run_config(run_id):
    """Return a run's saved settings. Keys added to BASE since then use their defaults."""
    config_path = Path(OUTPUT_DIR) / run_id / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Cannot resume, config not found: {config_path}")
    return {**BASE, **json.loads(config_path.read_text())}


def prepare_run_dir(cfg, run_id=None):
    """Create a run folder, or reopen an existing one when run_id is given."""
    is_resume = run_id is not None
    if not is_resume:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}_{cfg['exp']}"

    run_dir = Path(OUTPUT_DIR) / run_id
    if is_resume and not run_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume, run folder not found: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # A resumed run keeps its original config.json as the record of how it started.
    config_path = run_dir / "config.json"
    if not is_resume:
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return run_id, run_dir


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
