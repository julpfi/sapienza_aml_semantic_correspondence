"""Training: the last `unfreeze_blocks` DINOv2 blocks and the final LayerNorm (train_mode="finetune").
Hyperparameters: docs/RUN.md section 8.

Loss: cfg["loss"] over all accepted targets (src/models/loss.py), at loss_temperature (optionally
    annealed from loss_temperature_start over the first loss_temperature_epochs).
Data: train.csv minus the validation pairs (plus them with train_on_val), augmented by src/augment.py.
Batch: the optimiser sees `effective_batch` samples per update = micro-batch x accumulation steps.
    The micro-batch is the largest that fits the GPU (batch_size="auto", src/utils/batch_size.py).
Optimiser: AdamW (weight_decay) with OneCycleLR over all updates: warmup for warmup_epochs of them
    from max_lr / 25, then cosine down to max_lr / 25e4. One parameter group (own max_lr) per trained part.
pair_batch: one item = one image pair with all its queries (src/pair_batch.py), one backbone pass per
    image; batches and effective batch count pairs (pair_effective_batch), the loss of an update is the
    mean over its queries, and hard_neg_weight > 1 adds hard negatives (src/models/loss.py).
train_mode="lora": only the LoRA weights (src/models/lora.py); fusion="adapter" also trains model.fusion.
ema_decay > 0: an EMA of the trained weights (src/utils/ema.py) is validated and saved as best.pt.
Each epoch: validation PCK and its gain over the previous epoch against the 95% interval
(history.json, W&B), best.pt when it improves, last.pt for --resume, early stop after
`early_stop_patience` epochs without improvement.
Called from run.py: python run.py --mode train
"""

import json
import math
import random
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.datasets import QueryDataset, build_dataset, collate_samples, read_queries
from data.transforms import preprocess_image, to_resized
from src.augment import augment_pair, augment_sample, augmentation_enabled
from src.evaluate import evaluate
from src.metrics import target_group
from src.models.loss import LOGGED_STATS, LOSS_STATS, matching_loss, pair_matching_loss
from src.models.lora import lora_parameters
from src.pair_batch import PairDataset
from src.utils.batch_size import fake_samples, find_micro_batch
from src.utils.checkpoint import (
    checkpoint_path, download_artifact, load_checkpoint, load_model_weights, publish_best, save_checkpoint,
    trainable_state,
)
from src.utils.feature_store import FeatureStore
from src.utils.run import git_commit, seed_everything
from src.utils.wandb_log import init_wandb

TRAIN_MODES = ("frozen", "finetune", "lora", "sd_adapter")


# ---------------------------------------------------------------------------
# What is trained, with which learning rate
# ---------------------------------------------------------------------------
def configure_trainable(model, cfg):
    """Set requires_grad for the mode and return the optimiser parameter groups, "lr" = peak lr.

    Block i from the top: lr_backbone * layer_decay**i. Final LayerNorm: lr_backbone.
    Head (only runs whose config has head != "none"): lr_head.
    train_mode "lora": one group with the LoRA weights (lr_lora) instead of the blocks and LayerNorm.
    train_mode "sd_adapter": frozen pure-SD backbone + lightweight residual adapter (lr_adapter).
    fusion "adapter": one more group with model.fusion (lr_adapter). model.sd is never trained.
    """
    if cfg["train_mode"] not in TRAIN_MODES:
        raise ValueError(f"train_mode must be one of {TRAIN_MODES}, got {cfg['train_mode']!r}")
    if cfg["train_mode"] == "frozen" and cfg["fusion"] != "none":
        # The frozen path reads backbone features directly (and may cache them), so it would skip the fusion.
        raise ValueError("fusion needs train_mode 'finetune' or 'lora': the frozen path bypasses the SD features.")
    if model.backbone is not None:
        model.backbone.freeze()
        if cfg["train_mode"] == "finetune":
            model.backbone.unfreeze_last(cfg["unfreeze_blocks"])
    for p in model.head.parameters():
        p.requires_grad_(True)

    if isinstance(getattr(model, "sd", None), torch.nn.Module):
        model.sd.requires_grad_(False)

    parts = [("head", model.head, cfg["lr_head"])] if cfg["head"] != "none" else []
    if model.backbone is not None and cfg["train_mode"] != "lora":
        blocks = list(model.backbone.blocks)
        parts += [(f"block_{len(blocks) - 1 - depth}", block, cfg["lr_backbone"] * cfg["layer_decay"] ** depth)
                  for depth, block in enumerate(reversed(blocks))]
        parts.append(("layernorm", model.backbone.layernorm, cfg["lr_backbone"]))
    groups = [{"name": name, "params": [p for p in module.parameters() if p.requires_grad], "lr": lr}
              for name, module, lr in parts]
    groups = [g for g in groups if g["params"]]

    def add_group(name, params, lr):
        for p in params:
            p.requires_grad_(True)
        groups.append({"name": name, "params": params, "lr": lr})

    if cfg["train_mode"] == "lora":
        params = list(lora_parameters(model))
        if not params:
            raise ValueError("train_mode='lora' but the model has no LoRA parameters (apply_lora not called?).")
        if cfg.get("lora_train_layernorm"):
            for block in list(model.backbone.blocks)[-cfg["lora_blocks"]:]:
                for mod in block.modules():
                    if isinstance(mod, torch.nn.LayerNorm):
                        params.extend(mod.parameters())
            params.extend(model.backbone.layernorm.parameters())
        add_group("lora", params, cfg["lr_lora"])
    if cfg["train_mode"] == "sd_adapter":
        if not hasattr(model, "sd_adapter"):
            raise ValueError("train_mode='sd_adapter' requires backbone='sd'.")
        add_group("sd_adapter", list(model.sd_adapter.parameters()), cfg["lr_adapter"])
    if cfg["fusion"] in ("adapter", "adapter_mix"):
        add_group("adapter", list(model.fusion.parameters()), cfg["lr_adapter"])
    if cfg.get("cross_transform"):
        add_group("cross_transform", list(model.cross_transform.parameters()), cfg["lr_cross"])
    if not groups:
        raise ValueError("Nothing to train: train_mode='frozen' with head='none'. Use train_mode='finetune' "
                         "in BASE (src/config.py), or add head='linear' and lr_head for a head-only run.")
    return groups


# ---------------------------------------------------------------------------
# Batch sizes
# ---------------------------------------------------------------------------
def effective_batch_for(epoch, cfg):
    """Effective batch of an epoch: effective_batch, or the last ramp entry whose start epoch has passed.
    pair_batch: pair_effective_batch pairs (the ramp, counted in queries, is not used)."""
    if cfg["pair_batch"]:
        return cfg["pair_effective_batch"]
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


def updates_in_epochs(plan, epochs):
    """Optimiser updates in the first `epochs` epochs of the plan (fractions allowed)."""
    total = 0.0
    for _, updates in plan:
        total += updates * min(1.0, max(0.0, epochs))
        epochs -= 1
    return round(total)


# ---------------------------------------------------------------------------
# Forward pass and loss
# ---------------------------------------------------------------------------
def _backbone_features(model, samples, side, cfg, store):
    """Frozen backbone features [B, C, h, w] for one side ("src"/"trg"), using the disk store."""
    feats = [store.get(s[f"{side}_image"]) if store else None for s in samples]
    todo = [i for i, f in enumerate(feats) if f is None]
    if todo:
        x = torch.cat([preprocess_image(samples[i][f"{side}_img"], cfg) for i in todo]).to(model.device)
        with torch.no_grad():
            new = model.backbone(x)
        for i, f in zip(todo, new):
            feats[i] = f
            if store:
                store.put(samples[i][f"{side}_image"], f)
    return torch.stack([f.float().to(model.device) for f in feats])


def compute_features(model, samples, cfg, store):
    """Matching features (src, trg), each [B, C, h, w], in fp32, with gradients where needed."""
    if cfg["train_mode"] == "frozen":
        src = _backbone_features(model, samples, "src", cfg, store)
        trg = _backbone_features(model, samples, "trg", cfg, store)
        src, trg = model.head(src), model.head(trg)
    else:
        xs = torch.cat([preprocess_image(s["src_img"], cfg) for s in samples]).to(model.device)
        xt = torch.cat([preprocess_image(s["trg_img"], cfg) for s in samples]).to(model.device)
        src, trg = model(xs), model(xt)
    if hasattr(model, "cross_transform"):
        src, trg = model.pair_features(src, trg)
    return src, trg


def jitter_labels(trg_resized, image_size, n_cells, std_cells):
    """Gaussian noise on the target labels (GeoAware-SC regulariser), in patches."""
    if std_cells <= 0:
        return trg_resized
    noise = torch.randn(trg_resized.shape) * std_cells * image_size / n_cells
    return (trg_resized + noise).clamp(0, image_size)


def loss_temperature_at(update, ramp_updates, cfg):
    """Loss temperature of an optimiser update: loss_temperature, or with loss_temperature_start a
    linear ramp from it to loss_temperature over ramp_updates (ending exactly at loss_temperature)."""
    start, end = cfg["loss_temperature_start"], cfg["loss_temperature"]
    if not start or update >= ramp_updates - 1:
        return end
    return start + (end - start) * update / (ramp_updates - 1)


def batch_loss(model, samples, cfg, store, train=True, temperature=None):
    """The LOSS_STATS of src/models/loss.py per sample: three tensors [B], gradients on the first only."""
    src_feats, trg_feats = compute_features(model, samples, cfg, store)
    size = cfg["image_size"]
    stats = []
    for b, s in enumerate(samples):
        trg = to_resized(s["trg_xys"], s["trg_img"].size, size)
        if train:
            trg = jitter_labels(trg, size, trg_feats.shape[-1], cfg["label_jitter"])
        stats.append(matching_loss(src_feats[b], trg_feats[b],
                                   to_resized(s["src_xy"][None], s["src_img"].size, size), trg, cfg,
                                   temperature, scale_px=s.get("scale"), trg_wh=s["trg_img"].size))
    return tuple(torch.stack(values) for values in zip(*stats))


def pair_batch_loss(model, items, cfg, store, train=True, temperature=None):
    """batch_loss for a micro-batch of pairs (src/pair_batch.py): the LOSS_STATS of every query, three
    tensors [total queries] in item order, and the hard-negative cells per query [total queries]
    (None when hard negatives are off). Label jitter moves the own targets only, not the negatives."""
    src_feats, trg_feats = compute_features(model, items, cfg, store)
    size, w = cfg["image_size"], trg_feats.shape[-1]
    hard = cfg["hard_neg_weight"] > 1
    stats, n_negatives = [], []
    for b, item in enumerate(items):
        trg = [to_resized(xys, item["trg_img"].size, size) for xys in item["trg_xys"]]
        own = [jitter_labels(t, size, w, cfg["label_jitter"]) for t in trg] if train else trg
        src = to_resized(item["src_xys"], item["src_img"].size, size)
        item_stats, item_negatives = pair_matching_loss(src_feats[b], trg_feats[b], src, own, cfg, temperature,
                                                        neg_xys_resized=trg, return_negatives=True,
                                                        scales_px=item.get("scale"), trg_wh=item["trg_img"].size)
        stats.append(item_stats)
        if hard:
            n_negatives.append(item_negatives)
    return tuple(torch.cat(values) for values in zip(*stats)), torch.cat(n_negatives) if hard else None


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


def choose_micro_batch(model, cfg, groups):
    """batch_size: an int, or "auto" = probe the GPU with real forward+backward steps."""
    if cfg["batch_size"] != "auto":
        return int(cfg["batch_size"]), None
    rng = np.random.default_rng(0)

    def run_step(b):
        model.train()
        batch_loss(model, fake_samples(b, cfg, rng), cfg, None, train=False)[0].mean().backward()
        for g in groups:
            for p in g["params"]:
                p.grad = None

    if cfg["pair_batch"]:  # one pair = two images, like one fake sample
        max_batch = cfg["pair_effective_batch"]
    else:
        max_batch = (cfg["effective_batch"] if not cfg["effective_batch_schedule"]
                     else max(cfg["effective_batch_schedule"].values()))
    return find_micro_batch(run_step, max_batch, model.device)


@contextmanager
def averaged_weights(ema):
    """Inside the block the model holds the EMA weights; no-op when ema is None."""
    if ema is None:
        yield
        return
    ema.swap_in()
    try:
        yield
    finally:
        ema.swap_out()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg, model, run_id, run_dir, resume=False):
    """Train and return the best validation PCK (in percent)."""
    from src.diagnostics import log_evaluation, plot_worst_predictions

    pair_mode = bool(cfg["pair_batch"])
    if cfg["hard_neg_weight"] > 1 and not pair_mode:
        raise ValueError("hard_neg_weight > 1 needs pair_batch=True: the negatives come from the other "
                         "queries of the same pair.")
    groups = configure_trainable(model, cfg)
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"train_mode={cfg['train_mode']} head={cfg['head']}: {n_trainable / 1e6:.2f}M trainable parameters")

    ckpt = load_checkpoint(checkpoint_path(run_dir, "last.pt")) if resume else None
    if ckpt is not None and "micro_batch" in ckpt:
        micro, probe_peak = ckpt["micro_batch"], None  # same batches as before the disconnect
    else:
        micro, probe_peak = choose_micro_batch(model, cfg, groups)
    seed_everything(cfg["seed"], cfg["deterministic"])  # the probe used random numbers

    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    if cfg["train_on_val"]:
        dataset = (PairDataset if pair_mode else QueryDataset)(
            pd.concat([read_queries(cfg, "train"), read_queries(cfg, "val")]), cfg)
        print("train_on_val: training on the validation pairs too; no validation, best.pt = last epoch.")
    elif pair_mode:
        dataset = PairDataset(read_queries(cfg, "train"), cfg)
    else:
        dataset = build_dataset(cfg, "train")
    n_queries = len(dataset.df)
    validate = cfg["val_every_epoch"] and not cfg["train_on_val"]
    if len(dataset) == 0:
        raise ValueError("The training split is empty.")
    micro = min(micro, len(dataset))
    n_batches = len(dataset) // micro
    plan, total_updates = plan_updates(n_batches, micro, cfg)
    unit = "pairs" if pair_mode else "queries"
    if pair_mode:
        print(f"pair_batch: {len(dataset)} training pairs with {n_queries} queries "
              f"({n_queries / len(dataset):.1f} per pair), hard_neg_weight {cfg['hard_neg_weight']}")
    print(f"{len(dataset)} training {unit}, micro-batch {micro}, effective batch "
          f"{[micro * a for a, _ in plan]} {unit} per epoch, {total_updates} updates in {cfg['epochs']} epochs")
    # OneCycleLR divides by zero on its last step when the warmup covers every update (warmup_epochs >= epochs).
    warmup_updates = max(1, min(updates_in_epochs(plan, cfg["warmup_epochs"]), total_updates - 1))
    temperature_updates = updates_in_epochs(plan, cfg["loss_temperature_epochs"])
    print(f"lr warmup over {warmup_updates} updates, loss temperature ramp over {temperature_updates}")
    # Other OneCycleLR arguments at their defaults: div_factor 25, final_div_factor 1e4, cosine anneal.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups],
                                                    total_steps=total_updates,
                                                    pct_start=warmup_updates / total_updates)
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
        scheduler.load_state_dict(ckpt["scheduler"])
        _set_rng_state(ckpt["rng"])
        global_step, best, history = ckpt["global_step"], ckpt["best_score"], ckpt["history"]
        stale = ckpt.get("stale_epochs", 0)
        if ckpt["epoch_complete"]:
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch, skip_batches = ckpt["epoch"], ckpt["batches_done"]
        print(f"Resumed from last.pt: epoch {start_epoch}, skipping {skip_batches} batches, best {best:.2f}")
    elif resume and (run_dir / "history.json").exists() and checkpoint_path(run_dir, "best.pt").exists():
        raise RuntimeError(f"Run {run_id} already finished (last.pt is deleted at the end); start a new run.")
    elif resume:
        print("No last.pt yet: starting this run from the beginning.")
    elif cfg["init_artifact"]:
        init = load_checkpoint(download_artifact(cfg["init_artifact"], cfg["output_dir"]))
        load_model_weights(model, init["model"])
        print(f"Initialised from {cfg['init_artifact']}")
    ema_weights = None
    if cfg["ema_decay"] > 0:  # after every weight load, so the average starts from those weights
        from src.utils.ema import EMA

        ema_weights = EMA(model, cfg["ema_decay"])
        if ckpt is not None and "ema" in ckpt:
            ema_weights.load_state_dict(ckpt["ema"])

    wandb_run = init_wandb(cfg, run_id, run_dir, job_type="train")
    if wandb_run:
        wandb_run.summary.update({"micro_batch": micro, "total_updates": total_updates,
                                  "trainable_params_M": n_trainable / 1e6, "probe_peak_gb": probe_peak,
                                  "n_train_queries": n_queries,
                                  "cross_transform": bool(cfg.get("cross_transform"))})

    def save_last(epoch, batches_done, complete):
        save_checkpoint({
            "model": trainable_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "batches_done": batches_done, "epoch_complete": complete, "micro_batch": micro,
            "global_step": global_step, "best_score": best, "history": history, "stale_epochs": stale,
            "rng": _rng_state(), "config": cfg, "git_commit": git_commit(),
            **({"ema": ema_weights.state_dict()} if ema_weights else {}),
        }, run_dir, "last.pt")

    key = f"pck@{cfg['pck_alpha']}"
    val_set = build_dataset(cfg, "val") if wandb_run and validate else None
    ema = {}
    for epoch in range(start_epoch, cfg["epochs"]):
        accum, _ = plan[epoch]
        # The shuffle depends only on (seed, epoch), so a resumed epoch sees the same batches.
        loader = DataLoader(dataset, batch_size=micro, shuffle=True, drop_last=True,
                            num_workers=cfg["num_workers"], collate_fn=collate_samples,
                            generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_stats = defaultdict(list)  # per-sample values of the current update
        update_queries = 0  # pair_batch: queries whose summed loss is in the gradients
        t_epoch = time.time()
        progress = tqdm(loader, desc=f"epoch {epoch}", total=n_batches)
        for i, samples in enumerate(progress):
            if i < skip_batches:
                continue
            temperature = loss_temperature_at(global_step, temperature_updates, cfg)
            if pair_mode:
                group_of = {q: target_group(xys) for s in samples for q, xys in zip(s["query_ids"], s["trg_xys"])}
                if augment:
                    samples = [augment_pair(s, cfg) for s in samples]
                groups_of = [group_of[q] for s in samples for q in s["query_ids"]]
                stats, n_negatives = pair_batch_loss(model, samples, cfg, store, temperature=temperature)
            else:
                groups_of = [target_group(s["trg_xys"]) for s in samples]  # before augmentation drops targets
                if augment:
                    samples = [augment_sample(s, cfg) for s in samples]
                stats = batch_loss(model, samples, cfg, store, temperature=temperature)
            loss = stats[0].mean()
            if torch.isfinite(loss):
                if pair_mode:  # the sum now, divided by the update's query count before the step
                    stats[0].sum().backward()
                    update_queries += len(stats[0])
                    if n_negatives is not None:
                        step_stats["hard_neg_cells"] += n_negatives.tolist()
                else:
                    (loss / accum).backward()
                logged = dict(zip(LOSS_STATS, stats))
                for name in LOGGED_STATS:
                    step_stats[name] += logged[name].tolist()
                for value, group in zip(logged["loss_over_chance"].tolist(), groups_of):
                    step_stats[f"loss_over_chance/{group}"].append(value)
            else:
                print(f"Warning: non-finite loss at epoch {epoch} batch {i}; batch skipped.")
            if (i + 1) % accum and i + 1 != n_batches:
                continue

            # One optimiser update.
            if pair_mode and update_queries:
                for g in groups:
                    for p in g["params"]:
                        if p.grad is not None:
                            p.grad.div_(update_queries)
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]],
                                                             cfg["grad_clip"]))
            optimizer.step()
            if ema_weights:
                ema_weights.update()
            optimizer.zero_grad(set_to_none=True)
            lrs = {f"lr/{g['name']}": g["lr"] for g in optimizer.param_groups}  # lr of this update
            scheduler.step()
            global_step += 1
            means = {name: float(np.mean(values)) for name, values in step_stats.items()}
            for name in LOGGED_STATS:  # 0.95 EMA: memory of ~20 updates
                if name in means:
                    ema[name] = 0.95 * ema.get(name, means[name]) + 0.05 * means[name]
            if wandb_run:
                wandb_run.log({**{f"train/{name}": value for name, value in means.items()},
                               **{f"train/{name}_ema": value for name, value in ema.items()},
                               **lrs,
                               **({"train/loss_temperature": temperature} if cfg["loss_temperature_start"] else {}),
                               "batch/effective": micro * accum,
                               **({"batch/queries": update_queries} if pair_mode else {}),
                               "optim/grad_norm": grad_norm,
                               "optim/grad_clipped": float(grad_norm > cfg["grad_clip"])}, step=global_step)
            progress.set_postfix(loss_over_chance=f"{means.get('loss_over_chance', float('nan')):.3f}")
            step_stats, update_queries = defaultdict(list), 0
            if cfg["checkpoint_every_steps"] and global_step % cfg["checkpoint_every_steps"] == 0:
                save_last(epoch, i + 1, complete=False)
        skip_batches = 0

        record = {"epoch": epoch, "global_step": global_step, "epoch_seconds": round(time.time() - t_epoch, 1)}
        with averaged_weights(ema_weights if validate else None):  # validation and best.pt: EMA weights
            if validate:
                summary, meter = evaluate(cfg, model, run_dir, save=False, verbose=False)
                gain_text = ""
                if history and key in history[-1]:
                    # Gain over the previous epoch, in units of the 95% interval: inside (-1, 1) = within noise.
                    gain = summary[key] - history[-1][key]
                    summary[f"{key}_gain"] = gain
                    summary[f"{key}_gain_over_ci95"] = gain / max(summary[f"{key}_ci95"], 1e-9)
                    gain_text = f" (gain {gain:+.2f} = {summary[f'{key}_gain_over_ci95']:+.2f} x ci95)"
                record.update(summary)
                print(f"epoch {epoch}: val {key} = {summary[key]:.2f} ± {summary[f'{key}_ci95']:.1f}{gain_text}, "
                      f"val loss_over_chance {summary['loss_over_chance']:.3f}, "
                      f"entropy_norm {summary['entropy_norm']:.3f}")
                if summary[key] > best:
                    best, stale = summary[key], 0
                    save_checkpoint({"model": trainable_state(model), "epoch": epoch, "score": best,
                                     "config": cfg, "git_commit": git_commit()}, run_dir, "best.pt")
                else:
                    stale += 1
                record["best_pck"] = best
                if wandb_run:
                    figure = plot_worst_predictions(cfg, model, val_set, meter.frame())
                    log_evaluation(wandb_run, summary, meter, step=global_step, figure=figure)
                    wandb_run.summary["best_pck"] = best
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        save_last(epoch, n_batches, complete=True)
        if validate and cfg["early_stop_patience"] and stale >= cfg["early_stop_patience"]:
            print(f"Early stop: no better val {key} for {stale} epochs (best {best:.2f}).")
            break

    if not validate:  # keep the final weights as "best"
        with averaged_weights(ema_weights):
            save_checkpoint({"model": trainable_state(model), "epoch": cfg["epochs"] - 1, "score": None,
                             "config": cfg, "git_commit": git_commit()}, run_dir, "best.pt")
    # Training is over, so last.pt (weights + optimiser state, ~3x best.pt) is no longer needed on Drive.
    checkpoint_path(run_dir, "last.pt").unlink(missing_ok=True)
    if wandb_run:
        wandb_run.summary["best_pck"] = best
        if cfg["publish_best"]:
            publish_best(run_dir, run_id, {"score": best, "git_commit": git_commit()})
        wandb_run.finish()
    return best
