"""Save, resume and share checkpoints.

Two levels, on purpose:
- last.pt and best.pt go to the run folder (Drive on Colab). Free Colab can disconnect at
  any time, so save these often. They stay private to whoever ran the training.
- publish_best() uploads best.pt to W&B once, so teammates can use it. Do not upload every
  epoch: each new file counts towards the W&B storage limit.
"""

import os
from pathlib import Path

from src.utils.paths import OUTPUT_DIR


def save_checkpoint(state, run_dir, name="last.pt"):
    """Write via a temp file so a disconnect cannot leave a half-written checkpoint."""
    import torch

    path = Path(run_dir) / "checkpoints" / name
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(run_dir, name="last.pt", map_location="cpu"):
    """Return the saved state, or None if this run has no checkpoint yet."""
    import torch

    path = Path(run_dir) / "checkpoints" / name
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location)


def publish_best(run_dir, run_id, metadata=None):
    """Upload best.pt to W&B as `<run_id>-best`. Needs an active wandb run.

    Put the validation score, epoch and git commit in metadata. Returns the artifact, or
    None if there is no best.pt. A failed upload raises an error; best.pt stays on disk.
    """
    import wandb

    path = Path(run_dir) / "checkpoints" / "best.pt"
    if not path.exists():
        print(f"Nothing to publish, not found: {path}")
        return None

    artifact = wandb.Artifact(f"{run_id}-best", type="model", metadata=metadata or {})
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)
    # Wait for the upload, so Colab cannot close the session halfway through.
    if not wandb.run.offline:
        artifact.wait()
    return artifact


def download_artifact(name):
    """Download a teammate's model and return the path of its best.pt.

    Name format: "<entity>/<project>/<run_id>-best:latest" (shown on the W&B artifact page).
    Files are kept in outputs/artifacts/, so a second call does not download them again.
    """
    import wandb

    folder = name.rsplit("/", 1)[-1].replace(":", "_")
    root = wandb.Api().artifact(name, type="model").download(root=str(OUTPUT_DIR / "artifacts" / folder))
    return Path(root) / "best.pt"
