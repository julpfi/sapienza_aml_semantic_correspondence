"""Prepare a Colab session. Run it from the notebook with %run, so it can use Colab Secrets
and mount Drive in the notebook process:

    %run scripts/colab_setup.py --branch v1

Steps (each one is safe to repeat):
1. Pull the latest code of the branch.
2. Install the pip packages of environment.yml. Colab does not use conda, and it already has
   a CUDA build of torch/torchvision, so those two are skipped.
3. Load the secrets (Colab Secrets, or a .env file in the repo) into the environment.
4. Mount Google Drive, so runs, weights and feature caches survive a disconnect.
5. Download the competition data (scripts/kaggle_io.py) and the DINOv2 weights
   (scripts/download_weights.py).
Paths are then resolved by src/config.py for the "colab" runtime.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_ON_COLAB = ("torch", "torchvision")  # keep Colab's CUDA builds


def run(command):
    print("$", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def pull(branch):
    run(["git", "fetch", "origin", branch])
    run(["git", "checkout", branch])
    run(["git", "pull", "--ff-only", "origin", branch])


def pip_requirements():
    """The pip: list of environment.yml, without the packages Colab already provides."""
    import yaml  # PyYAML is preinstalled on Colab

    env = yaml.safe_load((REPO_ROOT / "environment.yml").read_text())
    pip_list = next(d["pip"] for d in env["dependencies"] if isinstance(d, dict) and "pip" in d)
    keep = []
    for req in pip_list:
        name = req.split("==")[0].split(">=")[0].split("[")[0].strip().lower()
        if name not in SKIP_ON_COLAB:
            keep.append(req)
    return keep


def install():
    run([sys.executable, "-m", "pip", "install", "-q", *pip_requirements()])


def version_check():
    """Print installed vs environment.yml versions. Colab keeps its own torch on purpose (pip, not
    Miniconda: docs/PRD_finetuning.md section 8), so a torch difference is expected, others are not."""
    from importlib.metadata import PackageNotFoundError, version

    import yaml

    env = yaml.safe_load((REPO_ROOT / "environment.yml").read_text())
    pip_list = next(d["pip"] for d in env["dependencies"] if isinstance(d, dict) and "pip" in d)
    print(f"{'package':<18}{'environment.yml':<18}installed")
    for req in pip_list:
        name, _, pinned = req.partition("==")
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "missing"
        note = "" if installed == pinned or name in SKIP_ON_COLAB else "  <- differs"
        print(f"{name:<18}{pinned:<18}{installed}{note}")


def load_colab_secrets():
    sys.path.insert(0, str(REPO_ROOT))
    from src.utils.env import load_secrets

    found = load_secrets()
    print(f"Secrets available: {found or 'none'}")


def mount_drive():
    try:
        from google.colab import drive
    except ImportError:
        print("Not in Colab: Drive not mounted.")
        return
    drive.mount("/content/drive")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branch", default="v1")
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--no-data", action="store_true")
    parser.add_argument("--no-weights", action="store_true")
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    if not args.no_pull:
        pull(args.branch)
    if not args.no_install:
        install()
    version_check()
    load_colab_secrets()
    if not args.no_drive:
        mount_drive()
    if not args.no_data:
        run([sys.executable, "scripts/kaggle_io.py", "download"])
    if not args.no_weights:
        run([sys.executable, "scripts/download_weights.py"])
    print("Colab setup complete.")


if __name__ == "__main__":
    main()
