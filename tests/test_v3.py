"""Test suite of the v3 additions (docs/PLAN.md): one test per check script, plus the two project checks.

    python -m pytest tests -q

Each check script prints PASS/FAIL lines and exits non-zero on a failure; a test fails with that output.
CPU only, no training. check_sd_fusion.py needs the tiny Stable Diffusion test pipeline in the Hugging Face
cache (downloaded on its first run).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMANDS = {
    "compileall": ["-m", "compileall", "-q", "run.py", "src", "data", "scripts"],
    "submit_dry_run": ["run.py", "--mode", "submit", "--dry-run"],
    "check_augmentation": ["scripts/check_augmentation.py"],
    "check_pair_batch": ["scripts/check_pair_batch.py"],
    "check_lora_ema": ["scripts/check_lora_ema.py"],
    "check_sd_fusion": ["scripts/check_sd_fusion.py"],
    "compare_runs_self_test": ["scripts/compare_runs.py", "--self-test"],
}


@pytest.mark.parametrize("name", list(COMMANDS))
def test_check(name):
    done = subprocess.run([sys.executable, *COMMANDS[name]], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, f"{name} failed:\n{done.stdout[-4000:]}\n{done.stderr[-4000:]}"
