"""Save the model weights to a local folder, so runs work offline (Kaggle) and start faster.

    python scripts/download_weights.py          # model_id / sd_model_id and revisions from src/config.py
    python scripts/download_weights.py --dinov3 # only the DINOv3 ViT-B/16 of DINOV3_B16 (gated: needs HF_TOKEN)

DINOv2: the snapshot goes to <weights_dir>/<model name>, e.g. weights/dinov2-base.
Stable Diffusion (only when fusion != "none"): the files SDFeatures loads (fp16 UNet and VAE, text encoder,
tokenizer, scheduler, configs; about 3.3 GB) go to <weights_dir>/<sd model name>, e.g. weights/stable-diffusion-2-1,
and SD_COMPLETE_MARKER is written last. On Colab weights_dir is on Drive, so SD is downloaded once and reused by
every later session; a download cut by a disconnect resumes on the next run of this script.
When a folder exists (SD: with its marker), the model loads from it instead of the Hub. For Kaggle, upload
weights/ as a dataset and attach it (see RUN.md).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import DINOV3_B16, get_config  # noqa: E402
from src.models.backbone import local_model_dir  # noqa: E402
from src.models.sd import SD_COMPLETE_MARKER, local_sd_dir, local_sd_ready  # noqa: E402
from src.utils.env import load_secrets  # noqa: E402

SD_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "*/config.json", "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dinov3", action="store_true", help="download the DINOv3 backbone of DINOV3_B16 only")
    args = parser.parse_args(argv)
    cfg = get_config()
    if args.dinov3:
        cfg.update(DINOV3_B16)
    load_secrets()

    from huggingface_hub import snapshot_download

    from huggingface_hub.errors import GatedRepoError

    target = local_model_dir(cfg)
    try:
        path = snapshot_download(repo_id=cfg["model_id"], revision=cfg["model_revision"], local_dir=target,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
    except GatedRepoError:
        sys.exit(f"{cfg['model_id']} is gated and the HF_TOKEN account has no access yet. Open "
                 f"https://huggingface.co/{cfg['model_id']} logged in with that account and request access (wait "
                 "for 'accepted'); a fine-grained token also needs 'Read access to contents of all public gated "
                 "repos you can access'. To run DINOv2 meanwhile: ACTIVE_PRESET = None in src/config.py.")
    print(f"{cfg['model_id']}@{cfg['model_revision']} -> {path}")

    if args.dinov3 or cfg["fusion"] == "none":
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
