"""Start a W&B run for training or evaluation (one helper, so both use the same project and naming)."""

import os
import tempfile
from pathlib import Path


def init_wandb(cfg, run_id, run_dir, job_type="train"):
    """Return a wandb run, or None when W&B is disabled.

    Local W&B files: online runs keep them on the local disk (they are uploaded, so they need not fill
    Drive); offline runs keep them in <run_dir>/wandb/ so they can be synced later.
    """
    mode = cfg["wandb_mode"]
    if mode == "auto":
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    if mode == "disabled" or os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb

    local_dir = Path(run_dir) if mode == "offline" else Path(tempfile.gettempdir()) / "wandb_runs" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    tags = [f"preset:{cfg.get('preset', 'none')}", f"fusion:{cfg['fusion']}"]
    return wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"], id=run_id, name=run_id,
                      job_type=job_type, config=cfg, tags=tags, resume="allow", mode=mode, dir=str(local_dir),
                      # no system charts (GPU, CPU, disk...) and no console capture
                      settings=wandb.Settings(x_disable_stats=True, console="off"))
