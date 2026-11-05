"""Training: the head with a frozen backbone (train_mode="frozen"), or the head plus the last
`unfreeze_blocks` DINOv2 blocks (train_mode="finetune"). Hyperparameters: docs/PRD_finetuning.md section 6.

Loss: Lang's soft-label matching loss, with all accepted targets (src/models/loss.py).
Data: train.csv minus the validation pairs, augmented by src/augment.py.
Batch: the optimiser sees `effective_batch` samples per update = micro-batch x accumulation steps.
    The micro-batch is the largest that fits the GPU (batch_size="auto", src/utils/batch_size.py).
LR: one parameter group per trained block (layer-wise decay), linear warmup then cosine, set every update.
Each epoch: validation PCK (history.json, W&B), best.pt when it improves, last.pt for --resume,
early stop after `early_stop_patience` epochs without improvement.
Called from run.py: python run.py --mode train
"""

import json
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.datasets import build_dataset, collate_samples
from data.transforms import preprocess_image, to_resized
from src.augment import augment_sample, augmentation_enabled
from src.evaluate import evaluate
from src.metrics import target_group
from src.models.loss import embedding_variance, matching_loss
from src.utils.batch_size import fake_samples, find_micro_batch
from src.utils.checkpoint import (
    checkpoint_path, download_artifact, load_checkpoint, load_model_weights, publish_best, save_checkpoint,
    trainable_state,
)
from src.utils.feature_store import FeatureStore
from src.utils.run import git_commit, seed_everything
from src.utils.wandb_log import init_wandb

TRAIN_MODES = ("frozen", "finetune")


# ---------------------------------------------------------------------------
# What is trained, with which learning rate
# ---------------------------------------------------------------------------
def _split_decay(name, params, lr, weight_decay):
    """Two groups: weights with weight decay, LayerNorm weights and biases (1-d) without."""
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    return [g for g in ({"name": name, "params": decay, "base_lr": lr, "weight_decay": weight_decay},
                        {"name": f"{name}_no_decay", "params": no_decay, "base_lr": lr, "weight_decay": 0.0})
            if g["params"]]


def configure_trainable(model, cfg):
    """Set requires_grad for the mode and return the optimiser parameter groups.

    Block i from the top gets lr_backbone * layer_decay**i; the final LayerNorm gets lr_backbone.
    """
    if cfg["train_mode"] not in TRAIN_MODES:
        raise ValueError(f"train_mode must be one of {TRAIN_MODES}, got {cfg['train_mode']!r}")
    model.backbone.freeze()
    if cfg["train_mode"] == "finetune":
        model.backbone.unfreeze_last(cfg["unfreeze_blocks"])
    head_params = list(model.head.parameters())
    for p in head_params:
        p.requires_grad_(True)

    wd = cfg["weight_decay"]
    groups = _split_decay("head", head_params, cfg["lr_head"], wd)
    blocks = list(model.backbone.blocks)
    for depth, block in enumerate(reversed(blocks)):
        params = [p for p in block.parameters() if p.requires_grad]
        if params:
            groups += _split_decay(f"block_{len(blocks) - 1 - depth}", params,
                                   cfg["lr_backbone"] * cfg["layer_decay"] ** depth, wd)
    norm = [p for p in model.backbone.layernorm.parameters() if p.requires_grad]
    groups += _split_decay("layernorm", norm, cfg["lr_backbone"], wd)
    if not groups:
        raise ValueError("Nothing to train: head='none' with train_mode='frozen'. In BASE (src/config.py), "
                         "set head='linear' (or 'mlp'), optionally with train_mode='finetune'.")
    for g in groups:
        g["lr"] = g["base_lr"]
    return groups


def lr_factor(step, total_steps, cfg):
    """Learning-rate multiplier at optimiser update `step` (0-based): linear warmup, then cosine or constant."""
    warmup = min(max(cfg["min_warmup_steps"], round(cfg["warmup_pct"] * total_steps)), max(total_steps - 1, 1))
    if step < warmup:
        return (step + 1) / warmup
    if cfg["lr_schedule"] == "constant":
        return 1.0
    if cfg["lr_schedule"] != "cosine":
        raise ValueError(f"lr_schedule must be 'cosine' or 'constant', got {cfg['lr_schedule']!r}")
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    floor = cfg["min_lr_ratio"]
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, factor):
    for g in optimizer.param_groups:
        g["lr"] = g["base_lr"] * factor


# ---------------------------------------------------------------------------
# Batch sizes
# ---------------------------------------------------------------------------
def effective_batch_for(epoch, cfg):
    """Effective batch of an epoch: effective_batch, or the last ramp entry whose start epoch has passed."""
    schedule = cfg["effective_batch_schedule"]
    if not schedule:
        return cfg["effective_batch"]
    started = [(int(e), b) for e, b in schedule.items() if int(e) <= epoch]  # JSON keys are strings
    return max(started)[1] if started else cfg["effective_batch"]


def plan_updates(n_batches, micro, cfg):
    """[(accumulation steps, updates)] per epoch, and the total number of optimiser updates."""
    plan = []
    for epoch in range(cfg["epochs"]):
        accum = max(1, math.ceil(effective_batch_for(epoch, cfg) / micro))
        plan.append((accum, math.ceil(n_batches / accum)))
    return plan, sum(updates for _, updates in plan)


# ---------------------------------------------------------------------------
# Forward pass and loss
# ---------------------------------------------------------------------------
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


def jitter_labels(trg_resized, image_size, n_cells, std_cells):
    """Gaussian noise on the target labels (GeoAware-SC regulariser), in patches."""
    if std_cells <= 0:
        return trg_resized
    noise = torch.randn(trg_resized.shape) * std_cells * image_size / n_cells
    return (trg_resized + noise).clamp(0, image_size)


def batch_loss(model, samples, cfg, store, use_amp, train=True):
    """Per-sample losses [B] and detached diagnostics (REVIEW_finetuning 1.4, 1.11):
    softmax_entropy and peak_mass per sample [B]; embed_var_batch over the source descriptors of the
    batch and embed_var_cells over the cells of each target map (mean over the batch), as floats."""
    src_feats, trg_feats = compute_features(model, samples, cfg, store, use_amp)
    size = cfg["image_size"]
    losses, stats = [], []
    for b, s in enumerate(samples):
        trg = to_resized(s["trg_xys"], s["trg_img"].size, size)
        if train:
            trg = jitter_labels(trg, size, trg_feats.shape[-1], cfg["label_jitter"])
        loss, sample_stats = matching_loss(src_feats[b], trg_feats[b],
                                           to_resized(s["src_xy"][None], s["src_img"].size, size), trg, size,
                                           cfg["loss_temperature"], cfg["label_sigma"], cfg["peak_mass_window"])
        losses.append(loss)
        stats.append(sample_stats)
    return torch.stack(losses), {
        "softmax_entropy": torch.stack([st["entropy"] for st in stats]),
        "peak_mass": torch.stack([st["peak_mass"] for st in stats]),
        "embed_var_batch": float(embedding_variance(torch.stack([st["query"] for st in stats]))),
        "embed_var_cells": float(np.mean([float(embedding_variance(f.flatten(1).T)) for f in trg_feats])),
    }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------
def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def choose_micro_batch(model, cfg, groups, use_amp):
    """batch_size: an int, or "auto" = probe the GPU with real forward+backward steps."""
    if cfg["batch_size"] != "auto":
        return int(cfg["batch_size"]), None
    rng = np.random.default_rng(0)

    def run_step(b):
        model.train()
        losses, _ = batch_loss(model, fake_samples(b, cfg, rng), cfg, None, use_amp, train=False)
        losses.mean().backward()
        for g in groups:
            for p in g["params"]:
                p.grad = None

    return find_micro_batch(run_step, cfg["effective_batch"] if not cfg["effective_batch_schedule"]
                            else max(cfg["effective_batch_schedule"].values()), model.device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg, model, run_id, run_dir, resume=False):
    """Train and return the best validation PCK (in percent)."""
    from src.diagnostics import log_evaluation, plot_predictions, worst_queries

    groups = configure_trainable(model, cfg)
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"train_mode={cfg['train_mode']} head={cfg['head']}: {n_trainable / 1e6:.2f}M trainable parameters")
    use_amp = cfg["amp"] and model.device.type == "cuda"

    ckpt = load_checkpoint(checkpoint_path(run_dir, "last.pt")) if resume else None
    if ckpt is not None and "micro_batch" in ckpt:
        micro, probe_peak = ckpt["micro_batch"], None  # same batches as before the disconnect
    else:
        micro, probe_peak = choose_micro_batch(model, cfg, groups, use_amp)
    seed_everything(cfg["seed"], cfg["deterministic"])  # the probe used random numbers

    optimizer = torch.optim.AdamW(groups)
    dataset = build_dataset(cfg, "train")
    if len(dataset) == 0:
        raise ValueError("The training split is empty.")
    micro = min(micro, len(dataset))
    n_batches = len(dataset) // micro
    plan, total_updates = plan_updates(n_batches, micro, cfg)
    print(f"{len(dataset)} training queries, micro-batch {micro}, effective batch "
          f"{[micro * a for a, _ in plan]} per epoch, {total_updates} updates in {cfg['epochs']} epochs")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    augment = cfg["flip_augment"] or augmentation_enabled(cfg)
    store = None
    if cfg["cache_features_to_disk"] and cfg["train_mode"] == "frozen":
        if augmentation_enabled(cfg):
            print("cache_features_to_disk is ignored: augmented images change every epoch.")
        else:
            store = FeatureStore(cfg)

    start_epoch, skip_batches, global_step, best, history, stale = 0, 0, 0, -math.inf, [], 0
    if ckpt is not None:
        load_model_weights(model, ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng"])
        global_step, best, history = ckpt["global_step"], ckpt["best_score"], ckpt["history"]
        stale = ckpt.get("stale_epochs", 0)
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

    wandb_run = init_wandb(cfg, run_id, run_dir, job_type="train")
    if wandb_run:
        wandb_run.summary.update({"micro_batch": micro, "total_updates": total_updates,
                                  "trainable_params_M": n_trainable / 1e6, "probe_peak_gb": probe_peak,
                                  "n_train_queries": len(dataset)})

    def save_last(epoch, batches_done, complete):
        save_checkpoint({
            "model": trainable_state(model), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "batches_done": batches_done, "epoch_complete": complete, "micro_batch": micro,
            "global_step": global_step, "best_score": best, "history": history, "stale_epochs": stale,
            "rng": _rng_state(), "config": cfg, "git_commit": git_commit(),
        }, run_dir, "last.pt")

    key = f"pck@{cfg['pck_alpha']}"
    val_set = build_dataset(cfg, "val") if wandb_run else None
    loss_ema = entropy_ema = None
    for epoch in range(start_epoch, cfg["epochs"]):
        accum, _ = plan[epoch]
        # The shuffle depends only on (seed, epoch), so a resumed epoch sees the same batches.
        loader = DataLoader(dataset, batch_size=micro, shuffle=True, drop_last=True,
                            num_workers=cfg["num_workers"], collate_fn=collate_samples,
                            generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        set_lr(optimizer, lr_factor(global_step, total_updates, cfg))
        step_losses, step_groups, step_stats = [], defaultdict(list), defaultdict(list)
        t_epoch = time.time()
        progress = tqdm(loader, desc=f"epoch {epoch}", total=n_batches)
        for i, samples in enumerate(progress):
            if i < skip_batches:
                continue
            groups_of = [target_group(s["trg_xys"]) for s in samples]  # before augmentation drops targets
            if augment:
                samples = [augment_sample(s, cfg) for s in samples]
            losses, stats = batch_loss(model, samples, cfg, store, use_amp)
            loss = losses.mean()
            if torch.isfinite(loss):
                scaler.scale(loss / accum).backward()
                step_losses += losses.detach().tolist()
                step_stats["softmax_entropy"] += stats["softmax_entropy"].tolist()
                step_stats["peak_mass"] += stats["peak_mass"].tolist()
                step_stats["embed_var_batch"].append(stats["embed_var_batch"])
                step_stats["embed_var_cells"].append(stats["embed_var_cells"])
                for value, group in zip(losses.detach().tolist(), groups_of):
                    step_groups[group].append(value)
            else:
                print(f"Warning: non-finite loss at epoch {epoch} batch {i}; batch skipped.")
            if (i + 1) % accum and i + 1 != n_batches:
                continue

            # One optimiser update.
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]],
                                                             cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            set_lr(optimizer, lr_factor(global_step, total_updates, cfg))
            mean_loss = float(np.mean(step_losses)) if step_losses else float("nan")
            means = {k: float(np.mean(v)) for k, v in step_stats.items()}
            if step_losses:  # 0.95 EMAs: memory of ~20 updates (REVIEW_finetuning 1.7)
                loss_ema = mean_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * mean_loss
                entropy = means["softmax_entropy"]
                entropy_ema = entropy if entropy_ema is None else 0.95 * entropy_ema + 0.05 * entropy
            if wandb_run:
                log = {"train/loss": mean_loss, "train/loss_ema": loss_ema, "train/softmax_entropy_ema": entropy_ema,
                       **{f"train/{k}": v for k, v in means.items()},
                       **{f"train/loss/{g}": float(np.mean(v)) for g, v in step_groups.items()},
                       **{f"lr/{g['name']}": g["lr"] for g in optimizer.param_groups
                          if not g["name"].endswith("_no_decay")},
                       "batch/effective": micro * accum,
                       "optim/grad_norm": grad_norm, "optim/grad_clipped": float(grad_norm > cfg["grad_clip"]),
                       "optim/amp_scale": scaler.get_scale() if use_amp else 1.0}
                wandb_run.log({k: v for k, v in log.items() if v is not None}, step=global_step)
            progress.set_postfix(loss=f"{mean_loss:.3f}")
            step_losses, step_groups, step_stats = [], defaultdict(list), defaultdict(list)
            if cfg["checkpoint_every_steps"] and global_step % cfg["checkpoint_every_steps"] == 0:
                save_last(epoch, i + 1, complete=False)
        skip_batches = 0

        record = {"epoch": epoch, "global_step": global_step, "epoch_seconds": round(time.time() - t_epoch, 1)}
        if cfg["val_every_epoch"]:
            summary, meter = evaluate(cfg, model, run_dir, save=False, verbose=False)
            record.update(summary)
            print(f"epoch {epoch}: val {key} = {summary[key]:.2f} ± {summary[f'{key}_ci95']:.1f}, "
                  f"val loss {summary.get('loss', float('nan')):.3f}")
            if summary[key] > best:
                best, stale = summary[key], 0
                save_checkpoint({"model": trainable_state(model), "epoch": epoch, "score": best,
                                 "config": cfg, "git_commit": git_commit()}, run_dir, "best.pt")
            else:
                stale += 1
            record["best_pck"] = best
            if wandb_run:
                figure = plot_predictions(cfg, model, val_set, worst_queries(meter.frame(), cfg["n_visual_checks"]))
                log_evaluation(wandb_run, summary, meter, step=global_step, figure=figure)
                wandb_run.summary["best_pck"] = best
                import matplotlib.pyplot as plt

                plt.close("all")
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        save_last(epoch, n_batches, complete=True)
        if cfg["val_every_epoch"] and cfg["early_stop_patience"] and stale >= cfg["early_stop_patience"]:
            print(f"Early stop: no better val {key} for {stale} epochs (best {best:.2f}).")
            break

    if not cfg["val_every_epoch"]:  # keep the final weights as "best"
        save_checkpoint({"model": trainable_state(model), "epoch": cfg["epochs"] - 1, "score": None,
                         "config": cfg, "git_commit": git_commit()}, run_dir, "best.pt")
    if wandb_run:
        wandb_run.summary["best_pck"] = best
        if cfg["publish_best"]:
            publish_best(run_dir, run_id, {"score": best, "git_commit": git_commit()})
        wandb_run.finish()
    return best
