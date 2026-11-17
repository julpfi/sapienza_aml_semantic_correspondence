"""Save the model weights to a local folder, so runs work offline (Kaggle) and start faster.

    python scripts/download_weights.py          # model_id / sd_model_id and revisions from src/config.py

DINOv2: the snapshot goes to <weights_dir>/<model name>, e.g. weights/dinov2-base.
Stable Diffusion (only when fusion != "none"): the files SDFeatures loads (fp16 UNet and VAE, text encoder,
tokenizer, scheduler, configs; about 3.3 GB) go to <weights_dir>/<sd model name>, e.g. weights/stable-diffusion-2-1,
and SD_COMPLETE_MARKER is written last. On Colab weights_dir is on Drive, so SD is downloaded once and reused by
every later session; a download cut by a disconnect resumes on the next run of this script.
When a folder exists (SD: with its marker), the model loads from it instead of the Hub. For Kaggle, upload
weights/ as a dataset and attach it (see RUN.md).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config  # noqa: E402
from src.models.backbone import local_model_dir  # noqa: E402
from src.models.sd import SD_COMPLETE_MARKER, local_sd_dir, local_sd_ready  # noqa: E402
from src.utils.env import load_secrets  # noqa: E402

SD_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "*/config.json", "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"]


def main():
    cfg = get_config()
    load_secrets()

    from huggingface_hub import snapshot_download

    target = local_model_dir(cfg)
    path = snapshot_download(repo_id=cfg["model_id"], revision=cfg["model_revision"], local_dir=target,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
    print(f"{cfg['model_id']}@{cfg['model_revision']} -> {path}")

    if cfg["fusion"] == "none":
        return
    sd_dir = local_sd_dir(cfg)
    marker = sd_dir / SD_COMPLETE_MARKER
    if local_sd_ready(cfg):
        print(f"{cfg['sd_model_id']}@{cfg['sd_model_revision']} already in {sd_dir}")
        return
    marker.unlink(missing_ok=True)
    path = snapshot_download(repo_id=cfg["sd_model_id"], revision=cfg["sd_model_revision"], local_dir=sd_dir,
                             allow_patterns=SD_FILES)
    marker.write_text(cfg["sd_model_revision"] + "\n")
    print(f"{cfg['sd_model_id']}@{cfg['sd_model_revision']} -> {path}")


if __name__ == "__main__":
    main()
