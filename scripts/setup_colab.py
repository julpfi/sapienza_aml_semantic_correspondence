"""Install dependencies and download the competition data. Safe to run repeatedly.

The notebook handles clone/pull, because this file only exists after cloning.

Needs KAGGLE_USERNAME and KAGGLE_KEY in the environment, loaded from Colab Secrets by
the notebook. Never hard-code keys here. Each teammate must also accept the competition
rules on the Kaggle website, otherwise the download fails with a 403 error.
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

from src.config import COMPETITION  # noqa: E402
from src.utils.paths import RAW_DATA_DIR  # noqa: E402

DONE_MARKER = RAW_DATA_DIR / ".download_complete"
OFFICIAL_FILES = ["train.csv", "test.csv", "make_submission.py", "sample_submission.csv"]


def install_requirements():
    print("Installing requirements...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_ROOT / "requirements.txt")],
        check=True,
    )


def download_data():
    if DONE_MARKER.exists():
        print(f"Data already present: {RAW_DATA_DIR}")
        return

    missing = [key for key in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.environ.get(key)]
    if missing:
        sys.exit(f"Missing {missing}. Add them to Colab Secrets and rerun the notebook.")

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        sys.exit("Kaggle CLI not found. Run without --skip-install.")

    print(f"Downloading {COMPETITION}...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [kaggle, "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DATA_DIR)]
    )
    if result.returncode != 0:
        sys.exit("Kaggle download failed. Have you accepted the competition rules?")

    archives = list(RAW_DATA_DIR.glob("*.zip"))
    if not archives:
        sys.exit(f"Kaggle reported success, but no zip file is in {RAW_DATA_DIR}.")
    for archive in archives:
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(RAW_DATA_DIR)
        archive.unlink()

    # Written last, so an interrupted download is retried on the next run.
    DONE_MARKER.touch()
    print(f"Data ready: {RAW_DATA_DIR}")


def report_official_files():
    for name in OFFICIAL_FILES:
        matches = list(RAW_DATA_DIR.rglob(name))
        status = matches[0] if matches else "not found in the download"
        print(f"  {name}: {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements()
    download_data()
    report_official_files()
    print("Setup complete.")


if __name__ == "__main__":
    main()
