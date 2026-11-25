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
from pathlib import Path

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
    parser.add_argument("--preset", default=None, help="Apply a named config preset from src/config.py (new runs)")
    parser.add_argument("--image-size", type=int, default=None, help="Override cfg['image_size'] for evaluate/submit")
    parser.add_argument("--soft-window", type=int, default=None, help="Override cfg['soft_window']")
    parser.add_argument("--soft-temperature", type=float, default=None, help="Override cfg['soft_temperature']")
    parser.add_argument("--flip-test", action=argparse.BooleanOptionalAction, default=None,
                        help="Override cfg['flip_test'] for evaluate/submit")
    parser.add_argument("--object-zoom", action=argparse.BooleanOptionalAction, default=None,
                        help="Override cfg['object_zoom'] for evaluate/submit")
    parser.add_argument("--cross-transform", action=argparse.BooleanOptionalAction, default=None,
                        help="Override cfg['cross_transform'] for new runs")
    parser.add_argument("--fusion", default=None, help="Override cfg['fusion'] for new runs")
    parser.add_argument("--val-fold", type=int, default=None,
                        help="Image-disjoint validation fold (0..4; -1 = legacy val_pair_ids.json)")
    parser.add_argument("--csls-weight", type=float, default=None,
                        help="Override cfg['csls_weight'] (CSLS hubness suppression; 0.0 = off)")
    parser.add_argument("--ensemble", default=None,
                        help="Comma-separated run IDs to ensemble in similarity space (evaluate/submit)")
    parser.add_argument("--ensemble-weights", default=None,
                        help="Comma-separated weights for --ensemble (default: equal)")
    parser.add_argument("--submission-name", default="submission.csv",
                        help="Filename for --mode submit inside <run_dir>/ (default: submission.csv)")
    args = parser.parse_args(argv)

    if args.resume and args.mode != "train":
        parser.error("--resume is only for --mode train (use --run for the other modes).")
    if args.run and args.mode == "train":
        parser.error("--run is for evaluate/submit; continue training with --resume.")
    if args.ensemble and args.mode == "train":
        parser.error("--ensemble is only for evaluate/submit.")
    return args


def _apply_cli_overrides(cfg, args):
    old_img_size = int(cfg["image_size"])
    old_sd_grid = int(cfg.get("sd_out_grid", 46))
    for attr, key in (("image_size", "image_size"), ("soft_window", "soft_window"),
                      ("soft_temperature", "soft_temperature"), ("flip_test", "flip_test"),
                      ("object_zoom", "object_zoom"), ("csls_weight", "csls_weight")):
        val = getattr(args, attr)
        if val is not None:
            cfg[key] = val
    # Architecture flags change the model, so a saved run keeps its own: its weights fit only that model.
    new_run = not (args.resume or args.run or args.ensemble)
    for attr in ("cross_transform", "fusion"):
        val = getattr(args, attr)
        if val is None:
            continue
        if new_run:
            cfg[attr] = val
        elif val != cfg.get(attr):
            print(f"Warning: --{attr.replace('_', '-')} {val} ignored; the saved run uses {cfg.get(attr)!r}.")
    if args.image_size is not None and cfg.get("backbone") == "sd":
        new_size = int(args.image_size)
        cfg["sd_image_size"] = new_size
        cfg["sd_out_grid"] = max(1, round(old_sd_grid * new_size / max(old_img_size, 1)))
    if args.val_fold is not None:
        if args.val_fold < 0:
            cfg["val_fold"] = None
        else:
            cfg["val_fold"] = args.val_fold
            cfg["max_val_queries"] = 0
    return cfg


def load_config(args):
    """The config of a new run (BASE), or the saved config of an existing run."""
    cfg = get_config(preset=None if (args.resume or args.run or args.ensemble) else args.preset)
    run_id = args.resume or args.run
    if args.ensemble:
        run_ids = [r.strip() for r in args.ensemble.split(",") if r.strip()]
        if run_id and run_id not in run_ids:
            run_ids = [run_id, *run_ids]
        cfg = load_run_config(run_ids[0], cfg["output_dir"])
        cfg["ensemble_runs"] = run_ids
        if args.ensemble_weights:
            cfg["ensemble_weights"] = [float(x) for x in args.ensemble_weights.split(",")]
        return _apply_cli_overrides(cfg, args), None
    if run_id:
        cfg = load_run_config(run_id, cfg["output_dir"])
        if cfg.get("git_commit") not in (None, git_commit()):
            print(f"Note: run {run_id} started on {cfg['git_commit']}, code is now {git_commit()}.")
    else:
        cfg["git_commit"] = git_commit()
        run_id = None
    return _apply_cli_overrides(cfg, args), run_id


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
    if args.ensemble:
        from src.models.ensemble import EnsembleModel
        from src.utils.run import run_dir_for

        members = []
        for member_id in cfg["ensemble_runs"]:
            mcfg = _apply_cli_overrides(load_run_config(member_id, cfg["output_dir"]), args)
            m = load_trained_weights(mcfg, build_model(mcfg, device), run_dir_for(cfg, member_id)).eval()
            members.append(m)
        model = EnsembleModel(members, cfg, weights=cfg.get("ensemble_weights"))
    else:
        model = build_model(cfg, device)

    if args.mode == "train":
        from src.train import train

        best = train(cfg, model, run_id, run_dir, resume=bool(args.resume))
        score = f"Best validation PCK@{cfg['pck_alpha']}: {best:.2f}" if best > float("-inf") else "No validation"
        print(f"{score}. Checkpoints in {run_dir / 'checkpoints'}")
        return 0

    if args.run and not args.ensemble:
        model = load_trained_weights(cfg, model, run_dir)

    if args.mode == "evaluate":
        from src.diagnostics import log_evaluation
        from src.evaluate import evaluate
        from src.models.matching import FeatureCache
        from src.sweep import describe, run_sweep
        from src.utils.wandb_log import init_wandb

        # One cache for evaluate and the sweep: the sweep reuses the features of the current setting.
        cache = FeatureCache(max(cfg["feature_cache_size"], 4096))
        summary, meter = evaluate(cfg, model, run_dir, with_loss=not args.ensemble, cache=cache)
        print(meter.per_group().to_string(index=False))
        print(meter.error_types().to_string())
        sweep = None if args.ensemble else run_sweep(cfg, model, device, cache=cache)
        if sweep is not None:
            sweep.to_csv(run_dir / "sweep.csv", index=False)
            print(describe(sweep, cfg))
        wandb_run = init_wandb(cfg, run_id, run_dir, job_type="evaluate") if cfg["wandb_evaluate"] else None
        if wandb_run:
            log_evaluation(wandb_run, summary, meter, sweep=sweep)
            wandb_run.finish()
        return 0

    from src.predict import check_submission, predict, write_submission

    predictions = predict(cfg, model, run_dir)
    sub_name = Path(args.submission_name).name or "submission.csv"
    if not sub_name.endswith(".csv"):
        sub_name = f"{sub_name}.csv"
    out = run_dir / sub_name
    write_submission(cfg, predictions, out)
    print(json.dumps(check_submission(cfg, out), indent=2))
    if cfg["runtime"] == "kaggle":  # a Kaggle notebook submits this file
        shutil.copy(out, cfg["kaggle_submission_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
