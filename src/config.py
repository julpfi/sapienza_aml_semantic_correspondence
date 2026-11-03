"""Run configuration: shared defaults plus named experiments.

Usage: python -m src.train --exp baseline --set lr=1e-4 batch_size=8
"""

import ast
from copy import deepcopy

COMPETITION = "aml-challenge-2026-27"

# Every setting must exist here. Unknown keys are rejected to catch typos.
# Model and training values are placeholders until the pipeline exists.
BASE = {
    "seed": 1019,

    # W&B
    "wandb_project": "sapienza-aml-semantic-correspondence",
    "wandb_entity": "pfingsten-sapienza-universit-di-roma",
    "publish_best": True,     # upload best.pt to W&B when training ends
    "init_artifact": None,  # start from a teammate's model: "<entity>/<project>/<run_id>-best:latest"

    # Model
    "model": "dino_baseline",
    "image_size": 512,
    
    # Training
    "batch_size": 8,
    "lr": 1e-4,
    "epochs": 10,
}

# Only list what differs from BASE.
EXPERIMENTS = {
    "baseline": {},
}


def parse_overrides(items):
    """Turn ["lr=1e-4", "model=foo"] into {"lr": 0.0001, "model": "foo"}."""
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        try:
            overrides[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            overrides[key] = raw  # plain strings need no quotes
    return overrides


def get_config(exp, overrides=None):
    """Merge BASE, the named experiment and command-line overrides."""
    if exp not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {exp!r}. Available: {sorted(EXPERIMENTS)}")

    cfg = deepcopy(BASE)
    for source in (EXPERIMENTS[exp], overrides or {}):
        unknown = set(source) - set(BASE)
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}. Add them to BASE first.")
        cfg.update(source)

    cfg["exp"] = exp
    return cfg
