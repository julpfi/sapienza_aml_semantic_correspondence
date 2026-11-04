"""Training: the head with a frozen backbone (train_mode="frozen"), or the head plus the last
`unfreeze_blocks` DINOv2 blocks (train_mode="finetune", Phase 2 on a Colab GPU).

Loss: Lang's soft-label matching loss, with all accepted targets (src/models/loss.py).
Data: train.csv minus the validation pairs (splits/val_pair_ids.json).
Each epoch: last.pt (for resuming), validation PCK (in history.json), and best.pt when it improves.
Called from run.py: python run.py --mode train (with head="linear" or "mlp" in src/config.py)
"""

import json
import math
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.datasets import build_dataset, collate_samples
from data.transforms import flip_sample, preprocess_image, to_resized
from src.evaluate import evaluate
from src.models.loss import matching_loss
from src.utils.checkpoint import (
    checkpoint_path, download_artifact, load_checkpoint, load_model_weights, publish_best, save_checkpoint,
    trainable_state,
)
from src.utils.feature_store import FeatureStore
from src.utils.run import git_commit, seed_everything

TRAIN_MODES = ("frozen", "finetune")


def configure_trainable(model, cfg):
    """Set requires_grad for the mode and return the optimiser parameter groups."""
    if cfg["train_mode"] not in TRAIN_MODES:
        raise ValueError(f"train_mode must be one of {TRAIN_MODES}, got {cfg['train_mode']!r}")
    model.backbone.freeze()
    if cfg["train_mode"] == "finetune":
        model.backbone.unfreeze_last(cfg["unfreeze_blocks"])
    head_params = list(model.head.parameters())
    for p in head_params:
        p.requires_grad_(True)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    groups = []
    if head_params:
        groups.append({"name": "head", "params": head_params, "lr": cfg["lr_head"]})
    if backbone_params:
        groups.append({"name": "backbone", "params": backbone_params, "lr": cfg["lr_backbone"]})
    if not groups:
        raise ValueError("Nothing to train: head='none' with train_mode='frozen'. In BASE (src/config.py), "
                         "set head='linear' (or 'mlp'), optionally with train_mode='finetune'.")
    return groups


def init_wandb(cfg, run_id):
    mode = cfg["wandb_mode"]
    if mode == "auto":
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    if mode == "disabled":
        return None
    import wandb

    return wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"], id=run_id, name=run_id,
                      config=cfg, resume="allow", mode=mode)


def _backbone_features(model, samples, side, cfg, store, use_amp):
    """Frozen backbone features [B, C, h, w] for one side ("src"/"trg"), using the disk store."""
    feats = [store.get(s[f"{side}_image"]) if store else None for s in samples]
    todo = [i for i, f in enumerate(feats) if f is None]
    if todo:
        x = torch.cat([preprocess_image(samples[i][f"{side}_img"], cfg) for i in todo]).to(model.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            new = model.backbone(x).float()
        for i, f in zip(todo, new):
            feats[i] = f
            if store:
                store.put(samples[i][f"{side}_image"], f)
    return torch.stack([f.float().to(model.device) for f in feats])


def compute_features(model, samples, cfg, store, use_amp):
    """Matching features (src, trg), each [B, C, h, w], with gradients where needed."""
    if cfg["train_mode"] == "frozen":
        src = _backbone_features(model, samples, "src", cfg, store, use_amp)
        trg = _backbone_features(model, samples, "trg", cfg, store, use_amp)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            return model.head(src).float(), model.head(trg).float()
    xs = torch.cat([preprocess_image(s["src_img"], cfg) for s in samples]).to(model.device)
    xt = torch.cat([preprocess_image(s["trg_img"], cfg) for s in samples]).to(model.device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        return model(xs).float(), model(xt).float()


def batch_loss(model, samples, cfg, store, use_amp):
    src_feats, trg_feats = compute_features(model, samples, cfg, store, use_amp)
    size = cfg["image_size"]
    losses = [
        matching_loss(src_feats[b], trg_feats[b],
                      to_resized(s["src_xy"][None], s["src_img"].size, size),
                      to_resized(s["trg_xys"], s["trg_img"].size, size),
                      size, cfg["loss_temperature"], cfg["label_sigma"])
        for b, s in enumerate(samples)
    ]
    return torch.stack(losses).mean()


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(cfg, model, run_id, run_dir, resume=False):
    """Train and return the best validation PCK (in percent)."""
    seed_everything(cfg["seed"], cfg["deterministic"])
    groups = configure_trainable(model, cfg)
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"train_mode={cfg['train_mode']} head={cfg['head']}: {n_trainable / 1e6:.2f}M trainable parameters")

    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    dataset = build_dataset(cfg, "train")
    if len(dataset) == 0:
        raise ValueError("The training split is empty.")
    batch_size = min(cfg["batch_size"], len(dataset))
    n_batches = len(dataset) // batch_size
    accum = cfg["grad_accum_steps"]
    steps_per_epoch = math.ceil(n_batches / accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=max(1, steps_per_epoch * cfg["epochs"]),
        pct_start=cfg["warmup_pct"])
    use_amp = cfg["amp"] and model.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    store = FeatureStore(cfg) if cfg["cache_features_to_disk"] and cfg["train_mode"] == "frozen" else None

    start_epoch, skip_batches, global_step, best, history = 0, 0, 0, -math.inf, []
    ckpt = load_checkpoint(checkpoint_path(run_dir, "last.pt")) if resume else None
    if ckpt is not None:
        load_model_weights(model, ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng"])
        global_step, best, history = ckpt["global_step"], ckpt["best_score"], ckpt["history"]
        if ckpt["scheduler"]["total_steps"] == scheduler.total_steps:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:  # epochs changed in config.json: follow the new schedule from the current step
            print(f"LR schedule length changed ({ckpt['scheduler']['total_steps']} -> {scheduler.total_steps} steps).")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(global_step):
                    scheduler.step()
        if ckpt["epoch_complete"]:
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch, skip_batches = ckpt["epoch"], ckpt["batches_done"]
        print(f"Resumed from last.pt: epoch {start_epoch}, skipping {skip_batches} batches, best {best:.2f}")
    elif resume:
        print("No last.pt yet: starting this run from the beginning.")
    elif cfg["init_artifact"]:
        init = load_checkpoint(download_artifact(cfg["init_artifact"], cfg["output_dir"]))
        load_model_weights(model, init["model"])
        print(f"Initialised from {cfg['init_artifact']}")

    wandb_run = init_wandb(cfg, run_id)

    def save_last(epoch, batches_done, complete):
        save_checkpoint({
            "model": trainable_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "batches_done": batches_done, "epoch_complete": complete,
            "global_step": global_step, "best_score": best, "history": history,
            "rng": _rng_state(), "config": cfg, "git_commit": git_commit(),
        }, run_dir, "last.pt")

    key = f"pck@{cfg['pck_alpha']}"
    for epoch in range(start_epoch, cfg["epochs"]):
        # The shuffle depends only on (seed, epoch), so a resumed epoch sees the same batches.
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg["num_workers"], collate_fn=collate_samples,
                            generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"epoch {epoch}", total=n_batches)
        for i, samples in enumerate(progress):
            if i < skip_batches:
                continue
            if cfg["flip_augment"]:
                samples = [flip_sample(s) if random.random() < cfg["flip_augment_prob"] else s for s in samples]
            loss = batch_loss(model, samples, cfg, store, use_amp)
            if torch.isfinite(loss):
                scaler.scale(loss / accum).backward()
            else:
                print(f"Warning: non-finite loss at epoch {epoch} batch {i}; batch skipped.")
            if (i + 1) % accum == 0 or i + 1 == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if wandb_run:
                    wandb_run.log({"train/loss": loss.item(), "train/epoch": epoch,
                                   **{f"lr/{g['name']}": g["lr"] for g in optimizer.param_groups}},
                                  step=global_step)
                if cfg["checkpoint_every_steps"] and global_step % cfg["checkpoint_every_steps"] == 0:
                    save_last(epoch, i + 1, complete=False)
            progress.set_postfix(loss=f"{loss.item():.3f}")
        skip_batches = 0

        record = {"epoch": epoch, "global_step": global_step}
        if cfg["val_every_epoch"]:
            summary, _ = evaluate(cfg, model, run_dir, save=False, verbose=False)
            record.update(summary)
            print(f"epoch {epoch}: val {key} = {summary[key]:.2f}")
            if wandb_run:
                wandb_run.log({f"val/{k}": v for k, v in summary.items()}, step=global_step)
            if summary[key] > best:
                best = summary[key]
                save_checkpoint({"model": trainable_state(model), "epoch": epoch, "score": best,
                                 "config": cfg, "git_commit": git_commit()}, run_dir, "best.pt")
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        save_last(epoch, n_batches, complete=True)

    if not cfg["val_every_epoch"]:  # keep the final weights as "best"
        save_checkpoint({"model": trainable_state(model), "epoch": cfg["epochs"] - 1, "score": None,
                         "config": cfg, "git_commit": git_commit()}, run_dir, "best.pt")
    if wandb_run:
        if cfg["publish_best"]:
            publish_best(run_dir, run_id, {"score": best, "git_commit": git_commit()})
        wandb_run.finish()
    return best
