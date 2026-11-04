"""Save, resume and share checkpoints.

Two levels, on purpose:
- last.pt and best.pt go to the run folder (Drive on Colab). Free Colab can disconnect at
  any time, so save these often. They stay private to whoever ran the training.
- publish_best() uploads best.pt to W&B once, so teammates can use it. Do not upload every
  epoch: each new file counts towards the W&B storage limit.

A checkpoint holds only the trainable weights (head and unfrozen backbone blocks). The frozen
backbone always comes from the pinned Hugging Face revision.
"""

import os
from pathlib import Path


def checkpoint_path(run_dir, name="last.pt"):
    return Path(run_dir) / "checkpoints" / name


def save_checkpoint(state, run_dir, name="last.pt"):
    """Write via a temp file so a disconnect cannot leave a half-written checkpoint."""
    import torch

    path = checkpoint_path(run_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(path, map_location="cpu"):
    """Return the saved state, or None if the file does not exist."""
    import torch

    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def trainable_state(model):
    """State dict restricted to parameters that are trained (head + unfrozen blocks)."""
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def load_model_weights(model, weights):
    """Load a trainable_state() dict into a model. Unknown keys are an error."""
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected:
        raise KeyError(f"Checkpoint has keys the model does not know: {unexpected[:5]}")
    return model


def publish_best(run_dir, run_id, metadata=None):
    """Upload best.pt to W&B as `<run_id>-best`. Needs an active wandb run.

    Put the validation score, epoch and git commit in metadata. Returns the artifact, or
    None if there is no best.pt. A failed upload raises an error; best.pt stays on disk.
    """
    import wandb

    path = checkpoint_path(run_dir, "best.pt")
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


def download_artifact(name, output_dir):
    """Download a teammate's model and return the path of its best.pt.

    Name format: "<entity>/<project>/<run_id>-best:latest" (shown on the W&B artifact page).
    Files are kept in <output_dir>/artifacts/, so a second call does not download them again.
    """
    import wandb

    folder = name.rsplit("/", 1)[-1].replace(":", "_")
    root = wandb.Api().artifact(name, type="model").download(root=str(Path(output_dir) / "artifacts" / folder))
    return Path(root) / "best.pt"
