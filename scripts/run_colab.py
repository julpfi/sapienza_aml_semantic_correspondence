"""Check the GPU and secrets, then start training.

WANDB_API_KEY and HF_TOKEN come from Colab Secrets via the notebook. Never hard-code them.
Training runs as a separate process, so freshly pulled code needs no kernel restart.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def has_gpu():
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"]).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", help="Ignored with --resume: the run keeps its saved settings")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("--dry-run", action="store_true", help="Skip GPU/W&B checks and training")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        if not has_gpu() and not args.allow_cpu:
            sys.exit("No GPU found. In Colab: Runtime > Change runtime type > GPU.")
        if not os.environ.get("WANDB_API_KEY"):
            sys.exit("WANDB_API_KEY is missing. Add it to Colab Secrets and rerun the notebook.")
        if not os.environ.get("HF_TOKEN"):
            print("Warning: HF_TOKEN is missing. Gated models such as DINOv3 will not download.")

    command = [sys.executable, "-m", "src.train"]
    if args.resume:
        command += ["--resume", args.resume]
    elif args.exp:
        command += ["--exp", args.exp]
    if args.set:
        command += ["--set", *args.set]
    if args.dry_run:
        command.append("--dry-run")

    sys.exit(subprocess.run(command, cwd=REPO_ROOT).returncode)


if __name__ == "__main__":
    main()
