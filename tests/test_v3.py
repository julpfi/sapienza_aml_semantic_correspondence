"""Test suite of the project: the v3 check scripts, the two project checks and the build checks.

    python -m pytest tests -q

Each check script prints PASS/FAIL lines and exits non-zero on a failure; a test fails with that output.
CPU only, no training. check_sd_fusion.py needs the tiny Stable Diffusion test pipeline in the Hugging Face
cache (downloaded on its first run).

The project has no build system (it runs from source), so "the build" is: every file compiles to bytecode,
every module imports, and the installed environment matches environment.yml.
"""

import importlib
import importlib.metadata
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

COMMANDS = {
    "compileall": ["-m", "compileall", "-q", "-f", "run.py", "src", "data", "scripts", "tests"],
    "pip_check": ["-m", "pip", "check"],
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


def test_every_module_imports():
    names = ["run"]
    for package in ("src", "src.models", "src.utils", "data"):
        module = importlib.import_module(package)
        names += [m.name for m in pkgutil.iter_modules(module.__path__, package + ".") if not m.ispkg]
    assert len(names) > 20
    for name in names:
        importlib.import_module(name)


def test_environment_matches_pins():
    env = yaml.safe_load((REPO_ROOT / "environment.yml").read_text())
    pins = next(d["pip"] for d in env["dependencies"] if isinstance(d, dict) and "pip" in d)
    wrong = {}
    for pin in pins:
        name, version = pin.split("==")
        installed = importlib.metadata.version(name).split("+")[0]  # torch reports e.g. 2.15.0+cpu
        if installed != version:
            wrong[name] = (version, installed)
    assert not wrong, f"installed versions differ from environment.yml: {wrong}"
