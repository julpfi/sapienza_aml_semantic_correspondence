"""Training entry point. The run setup works; the training loop is still a TODO.

Usage:
    python -m src.train --exp baseline --set lr=1e-4
    python -m src.train --resume <run_id>   # settings come from the run's config.json
    python -m src.train --exp baseline --dry-run   # checks the setup only
"""

import argparse

from src.config import get_config, parse_overrides
from src.utils.paths import OUTPUT_DIR
from src.utils.run import git_commit, load_run_config, prepare_run_dir, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", help="Experiment name from src/config.py")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", metavar="RUN_ID", help="Continue a run in the output folder")
    parser.add_argument("--dry-run", action="store_true", help="Stop after the setup")
    args = parser.parse_args()

    if args.resume and args.set:
        parser.error("--set cannot be combined with --resume: a run keeps its saved settings.")
    if not args.resume and not args.exp:
        parser.error("Provide --exp for a new run or --resume for an existing one.")
    return args


def main():
    args = parse_args()
    if args.resume:
        cfg = load_run_config(args.resume)
        if git_commit() != cfg["git_commit"]:
            print(f"Warning: run started on {cfg['git_commit']}, code is now {git_commit()}.")
    else:
        cfg = get_config(args.exp, parse_overrides(args.set))
        cfg["git_commit"] = git_commit()

    if args.dry_run:
        print(f"Dry run OK. Runs would be saved to: {OUTPUT_DIR}")
        print(cfg)
        return

    run_id, run_dir = prepare_run_dir(cfg, args.resume)
    print(f"Run {run_id} -> {run_dir}")
    seed_everything(cfg["seed"])

    import wandb

    wandb.init(
        project=cfg["wandb_project"],
        entity=cfg["wandb_entity"],
        id=run_id,
        name=run_id,
        config=cfg,
        resume="allow",
    )

    # TODO: build the data loaders and model from cfg, then write the loop:
    # - On --resume, restore with load_checkpoint(run_dir).
    # - Otherwise, if cfg["init_artifact"] is set, load download_artifact(cfg["init_artifact"]).
    # - Log validation PCK to W&B.
    # - Each epoch, save_checkpoint(state, run_dir) writes last.pt for resuming.
    # - When validation PCK improves, save_checkpoint(state, run_dir, "best.pt").
    # - At the end, if cfg["publish_best"], call publish_best(run_dir, run_id, metadata).
    wandb.finish(exit_code=1)
    raise NotImplementedError("The training loop has not been written yet.")


if __name__ == "__main__":
    main()
