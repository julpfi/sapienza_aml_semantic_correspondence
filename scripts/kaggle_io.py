"""Download the competition data, or upload a submission, with the Kaggle CLI.

    python scripts/kaggle_io.py download                         # into data_dir (data/raw locally)
    python scripts/kaggle_io.py submit --file outputs/<run_id>/submission.csv --message "baseline"

Needs KAGGLE_USERNAME and KAGGLE_KEY (from .env or Colab Secrets). Each teammate must accept
the competition rules on the Kaggle website first, otherwise the download fails with a 403.
Safe to run repeatedly: a finished download is not repeated.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config  # noqa: E402
from src.utils.env import load_secrets  # noqa: E402

DONE_MARKER = ".download_complete"
OFFICIAL_FILES = ["train.csv", "test.csv", "sample_submission.csv", "make_submission.py", "metric.py"]


def kaggle_cli():
    load_secrets()
    # New Kaggle API tokens (KGAT_...) are read from KAGGLE_API_TOKEN, not KAGGLE_KEY: with the token in
    # KAGGLE_KEY the CLI sends it as a legacy key and the download fails with 401 Unauthorized.
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if key.startswith("KGAT_") and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = key
    if os.environ.get("KAGGLE_API_TOKEN"):
        os.environ.pop("KAGGLE_KEY", None)  # the CLI would otherwise prefer the username / key pair
    else:
        missing = [k for k in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.environ.get(k)]
        if missing:
            sys.exit(f"Missing {missing} (or KAGGLE_API_TOKEN). Put them in .env (local) or Colab Secrets.")
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        sys.exit("Kaggle CLI not found. Install the environment first (environment.yml).")
    return kaggle


def download(cfg):
    data_dir = Path(cfg["data_dir"])
    if (data_dir / DONE_MARKER).exists():
        print(f"Data already present: {data_dir}")
        return data_dir
    kaggle = kaggle_cli()
    print(f"Downloading {cfg['competition']} -> {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([kaggle, "competitions", "download", "-c", cfg["competition"], "-p", str(data_dir)])
    if result.returncode != 0:
        sys.exit("Kaggle download failed. Have you accepted the competition rules?")
    archives = list(data_dir.glob("*.zip"))
    if not archives:
        sys.exit(f"Kaggle reported success, but no zip file is in {data_dir}.")
    for archive in archives:
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(data_dir)
        archive.unlink()
    (data_dir / DONE_MARKER).touch()  # written last, so an interrupted download is retried
    for name in OFFICIAL_FILES:
        matches = list(data_dir.rglob(name))
        print(f"  {name}: {matches[0] if matches else 'not in the download'}")
    return data_dir


def submit(cfg, file, message):
    kaggle = kaggle_cli()
    result = subprocess.run([kaggle, "competitions", "submit", "-c", cfg["competition"], "-f", str(file),
                             "-m", message])
    if result.returncode != 0:
        sys.exit("Kaggle submission failed.")
    print("Submitted. Check the score: kaggle competitions submissions -c " + cfg["competition"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["download", "submit"])
    parser.add_argument("--file", help="submit: path of submission.csv")
    parser.add_argument("--message", default="", help="submit: description shown on Kaggle")
    args = parser.parse_args()
    cfg = get_config()
    if args.action == "download":
        download(cfg)
    else:
        if not args.file:
            parser.error("submit needs --file")
        submit(cfg, args.file, args.message)


if __name__ == "__main__":
    main()
