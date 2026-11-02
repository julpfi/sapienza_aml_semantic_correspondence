"""Stage 2: light supervised fine-tuning of the last DINO blocks on keypoint pairs.

Loss: for every annotated source keypoint, its feature is compared with all target patches;
a softmax over target patches is trained to put mass on the ground-truth location
(Gaussian soft label, sigma in patches). This is the standard "dense matching as classification" loss.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .backbone import DinoFeatures, preprocess
from .data import build_dataset, collate_keep
from .evaluate import evaluate
from .matching import cell_centers, sample_features


def _flip_sample(s):
    s = dict(s)
    for side in ("src", "trg"):
        w = s[f"{side}_img"].size[0]
        s[f"{side}_img"] = s[f"{side}_img"].transpose(Image.FLIP_LEFT_RIGHT)
        k = s[f"{side}_kps"].copy()
        k[:, 0] = w - k[:, 0]
        s[f"{side}_kps"] = k
    return s


def _to_resized(kps, wh, S):
    k = torch.as_tensor(kps, dtype=torch.float32).clone()
    k[:, 0] *= S / wh[0]
    k[:, 1] *= S / wh[1]
    return k


def matching_loss(fs, ft, src_k, trg_k, S, tau, sigma):
    """fs, ft [C,h,w] ; src_k, trg_k [K,2] resized coords."""
    C, h, w = ft.shape
    q = sample_features(fs, src_k.to(fs.device), S)                  # [K,C]
    logits = (q @ ft.flatten(1)) / tau                               # [K,h*w]
    cx, cy = cell_centers(w, S, ft.device), cell_centers(h, S, ft.device)
    patch = S / w
    if sigma > 0:
        dx = (cx[None, None, :] - trg_k[:, 0, None, None].to(ft.device)) / patch
        dy = (cy[None, :, None] - trg_k[:, 1, None, None].to(ft.device)) / patch
        target = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)).flatten(1)
        target = target / target.sum(-1, keepdim=True)
    else:
        col = (trg_k[:, 0] / patch).long().clamp(0, w - 1)
        row = (trg_k[:, 1] / patch).long().clamp(0, h - 1)
        target = F.one_hot(row * w + col, h * w).float().to(ft.device)
    return -(target * F.log_softmax(logits, -1)).sum(-1).mean()


def finetune(cfg, val_every_epoch=True):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = cfg.resolved_device()
    model = DinoFeatures(cfg).to(device)
    model.unfreeze_last(cfg.ft_unfreeze_blocks)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=cfg.ft_lr, weight_decay=0.05)
    train_ds = build_dataset(cfg.replace(max_pairs=cfg.ft_max_train_pairs), split="trn")
    loader = DataLoader(train_ds, batch_size=cfg.ft_batch_size, shuffle=True,
                        num_workers=cfg.num_workers, collate_fn=collate_keep, drop_last=True)
    steps = cfg.ft_epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.ft_lr, total_steps=steps, pct_start=0.05)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    S = cfg.img_size
    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for ep in range(cfg.ft_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {ep}")
        for batch in pbar:
            if cfg.ft_pair_flip_aug:
                batch = [_flip_sample(s) if random.random() < 0.5 else s for s in batch]
            xs = torch.cat([preprocess(s["src_img"], S) for s in batch]).to(device)
            xt = torch.cat([preprocess(s["trg_img"], S) for s in batch]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                FS, FT = model(xs), model(xt)
            FS, FT = FS.float(), FT.float()
            loss = sum(
                matching_loss(FS[b], FT[b], _to_resized(s["src_kps"], s["src_img"].size, S),
                              _to_resized(s["trg_kps"], s["trg_img"].size, S), S, cfg.ft_temperature, cfg.ft_label_sigma)
                for b, s in enumerate(batch)) / len(batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        trainable_sd = {k: v.detach().cpu() for k, v in model.model.state_dict().items()}
        torch.save(trainable_sd, out / "last.pth")
        if val_every_epoch:
            model.eval()
            summ, _ = evaluate(cfg.replace(split="val", run_name=f"{cfg.run_name}/val_ep{ep}"),
                               model=model, device=device, verbose=False)
            score = summ["pck@0.1 (per-kp)"]
            history.append({"epoch": ep, **summ})
            print(f"epoch {ep}: val PCK@0.1 = {score:.2f}")
            if score > best:
                best = score
                torch.save(trainable_sd, out / "best.pth")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return best
