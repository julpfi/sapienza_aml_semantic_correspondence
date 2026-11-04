"""Save the DINOv2 weights to a local folder, so runs work offline (Kaggle) and start faster.

    python scripts/download_weights.py          # model_id/model_revision from src/config.py

The snapshot goes to <weights_dir>/<model name>, e.g. weights/dinov2-base. When that folder
exists, the backbone loads from it instead of the Hub. For Kaggle, upload weights/ as a
dataset and attach it (see RUN.md).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config  # noqa: E402
from src.models.backbone import local_model_dir  # noqa: E402
from src.utils.env import load_secrets  # noqa: E402


def main():
    cfg = get_config()
    load_secrets()

    from huggingface_hub import snapshot_download

    target = local_model_dir(cfg)
    path = snapshot_download(repo_id=cfg["model_id"], revision=cfg["model_revision"], local_dir=target,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
    print(f"{cfg['model_id']}@{cfg['model_revision']} -> {path}")


if __name__ == "__main__":
    main()
