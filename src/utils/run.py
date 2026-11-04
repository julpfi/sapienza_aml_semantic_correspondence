"""Run folders, saved configs and seeds."""

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.config import BASE, PATH_ENV_VARS, REPO_ROOT, resolve_paths


def git_commit():
    """Return the current commit, marked -dirty if there are uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def run_dir_for(cfg, run_id):
    return Path(cfg["output_dir"]) / run_id


def load_run_config(run_id, output_dir):
    """Return a run's saved settings. Keys added to BASE since then use their defaults."""
    config_path = Path(output_dir) / run_id / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    saved = json.loads(config_path.read_text())
    cfg = {**BASE, **saved}
    cfg["runtime"] = BASE["runtime"]
    for key, env_var in PATH_ENV_VARS.items():
        # Priority: SC_* variable, then the saved folder if it exists here, then the runtime default.
        if os.environ.get(env_var) or not (cfg.get(key) and Path(cfg[key]).is_dir()):
            cfg[key] = None
    cfg["output_dir"] = str(output_dir)
    return resolve_paths(cfg)


def prepare_run_dir(cfg, run_id=None, mode="train"):
    """Create <output_dir>/<stamp>_<mode>/ with config.json, or reopen an existing run."""
    is_existing = run_id is not None
    if not is_existing:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}_{mode}"

    run_dir = run_dir_for(cfg, run_id)
    if is_existing and not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # An existing run keeps its original config.json as the record of how it started.
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return run_id, run_dir


def seed_everything(seed, deterministic=True):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
