"""Start a W&B run for training or evaluation (one helper, so both use the same project and naming)."""

import os


def init_wandb(cfg, run_id, run_dir, job_type="train"):
    """Return a wandb run, or None when W&B is disabled. Its local files go to <run_dir>/wandb/."""
    mode = cfg["wandb_mode"]
    if mode == "auto":
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    if mode == "disabled" or os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb

    return wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"], id=run_id, name=run_id,
                      job_type=job_type, config=cfg, resume="allow", mode=mode, dir=str(run_dir),
                      # no system charts (GPU, CPU, disk...) and no console capture (REVIEW_finetuning 2.2)
                      settings=wandb.Settings(x_disable_stats=True, console="off"))
