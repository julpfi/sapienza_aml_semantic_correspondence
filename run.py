"""Single entry point for the pipeline. Every mode uses the same config, data and model code.

    python run.py --mode evaluate                     # validation PCK
    python run.py --mode train                        # train (set head / train_mode in src/config.py)
    python run.py --mode train    --resume <run_id>   # continue a training run after a disconnect
    python run.py --mode evaluate --run <run_id>      # evaluate a trained run (its best.pt)
    python run.py --mode submit   [--run <run_id>]    # test predictions + submission.csv
    python run.py --mode submit   --dry-run           # print settings and check files

Settings: BASE in src/config.py (the only configuration).
Results go to <output_dir>/<YYYYMMDD-HHMMSS>_<mode>/, always with config.json (see RUN.md).
"""

import argparse
import json
import shutil
import sys

from src.config import data_path, get_config, resolved_device
from src.utils.env import load_secrets
from src.utils.run import git_commit, load_run_config, prepare_run_dir

MODES = ("train", "evaluate", "submit")
NEEDED_FILES = {"train": ["train_csv"], "evaluate": ["train_csv"], "submit": ["test_csv", "sample_submission_csv"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--resume", metavar="RUN_ID", help="train: continue this run from last.pt")
    parser.add_argument("--run", metavar="RUN_ID", help="evaluate/submit: use this run's config and best.pt")
    parser.add_argument("--require-gpu", action="store_true", help="Stop if no CUDA GPU is available")
    parser.add_argument("--dry-run", action="store_true", help="Print the settings and check the files only")
    args = parser.parse_args(argv)

    if args.resume and args.mode != "train":
        parser.error("--resume is only for --mode train (use --run for the other modes).")
    if args.run and args.mode == "train":
        parser.error("--run is for evaluate/submit; continue training with --resume.")
    return args


def load_config(args):
    """The config of a new run (BASE), or the saved config of an existing run."""
    cfg = get_config()
    run_id = args.resume or args.run
    if run_id:
        cfg = load_run_config(run_id, cfg["output_dir"])
        if cfg.get("git_commit") not in (None, git_commit()):
            print(f"Note: run {run_id} started on {cfg['git_commit']}, code is now {git_commit()}.")
        return cfg, run_id
    cfg["git_commit"] = git_commit()
    return cfg, None


def preflight(cfg, args):
    """Check the device and the data files before a long run. Returns (problems, device)."""
    problems = []
    device = resolved_device(cfg)
    if args.require_gpu and device != "cuda":
        problems.append("No CUDA GPU found. In Colab: Runtime > Change runtime type > GPU.")
    for key in NEEDED_FILES[args.mode]:
        if not data_path(cfg, key).exists():
            problems.append(f"Missing {data_path(cfg, key)}. Download the data: python scripts/kaggle_io.py download")
    return problems, device


def load_trained_weights(cfg, model, run_dir):
    """Load best.pt of a trained run. An untrained baseline run has nothing to load."""
    from src.utils.checkpoint import checkpoint_path, load_checkpoint, load_model_weights

    path = checkpoint_path(run_dir, "best.pt")
    ckpt = load_checkpoint(path)
    if ckpt is None:
        if cfg["head"] == "none" and cfg["train_mode"] == "frozen":
            return model
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    print(f"Loaded weights from {path}")
    return load_model_weights(model, ckpt["model"])


def main(argv=None):
    args = parse_args(argv)
    load_secrets()
    cfg, run_id = load_config(args)
    problems, device = preflight(cfg, args)

    if args.dry_run:
        print(json.dumps({k: cfg[k] for k in ("runtime", "data_dir", "output_dir", "weights_dir", "splits_dir",
                                               "model_id", "model_revision", "image_size", "head",
                                               "train_mode")}, indent=2))
        print(f"device: {device}")
        print("\n".join(problems) if problems else "Dry run OK: all needed files are present.")
        return 1 if problems else 0
    if problems:
        sys.exit("\n".join(problems))

    from src.models import build_model
    from src.utils.run import seed_everything

    seed_everything(cfg["seed"], cfg["deterministic"])
    run_id, run_dir = prepare_run_dir(cfg, run_id, args.mode)
    print(f"Run {run_id} -> {run_dir} (settings in config.json)")
    model = build_model(cfg, device)

    if args.mode == "train":
        from src.train import train

        best = train(cfg, model, run_id, run_dir, resume=bool(args.resume))
        print(f"Best validation PCK@{cfg['pck_alpha']}: {best:.2f}. Checkpoints in {run_dir / 'checkpoints'}")
        return 0

    if args.run:
        model = load_trained_weights(cfg, model, run_dir)

    if args.mode == "evaluate":
        from data.datasets import build_dataset
        from src.diagnostics import log_evaluation, plot_predictions, worst_queries
        from src.evaluate import evaluate
        from src.models.matching import FeatureCache
        from src.sweep import describe, run_sweep
        from src.utils.wandb_log import init_wandb

        # One cache for evaluate and the sweep: the sweep reuses the features of the current setting.
        cache = FeatureCache(max(cfg["feature_cache_size"], 4096))
        summary, meter = evaluate(cfg, model, run_dir, cache=cache)
        print(meter.per_group().to_string(index=False))
        print(meter.error_types().to_string())
        sweep = run_sweep(cfg, model, device, cache=cache)
        if sweep is not None:
            sweep.to_csv(run_dir / "sweep.csv", index=False)
            print(describe(sweep, cfg))
        png = run_dir / "visual_check.png"
        figure = plot_predictions(cfg, model, build_dataset(cfg, "val"),
                                  worst_queries(meter.frame(), cfg["n_visual_checks"]), png)
        print(f"Visual check (worst queries): {png}")
        wandb_run = init_wandb(cfg, run_id, run_dir, job_type="evaluate") if cfg["wandb_evaluate"] else None
        log_evaluation(wandb_run, summary, meter, figure=figure, sweep=sweep)
        if wandb_run:
            wandb_run.finish()
        return 0

    from src.predict import check_submission, predict, write_submission

    predictions = predict(cfg, model, run_dir)
    out = run_dir / "submission.csv"
    write_submission(cfg, predictions, out)
    print(json.dumps(check_submission(cfg, out), indent=2))
    if cfg["runtime"] == "kaggle":  # a Kaggle notebook submits this file
        shutil.copy(out, cfg["kaggle_submission_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
