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

def positive_patch_mask(targets, target_wh, grid_hw, *, device,
                        radius_patches=1.0):
    """Union of patch neighborhoods; include the nearest cell for EACH target.

    Coordinates follow the pipeline's continuous [0, width]/[0, height]
    convention. Out-of-range annotations are rejected, not silently clamped.
    This native-grid surrogate is NOT the competition PCK acceptance region.
    """
    w_img, h_img = map(float, target_wh)
    h, w = map(int, grid_hw)
    if min(w_img, h_img, h, w) <= 0 or radius_patches < 0:
        raise ValueError("Invalid image/grid dimensions or radius")
    points = torch.as_tensor(targets, dtype=torch.float32, device=device)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("accepted_targets must be a nonempty [N, 2] array")
    if not torch.isfinite(points).all().item():
        raise ValueError("Non-finite accepted target")
    bounds = points.new_tensor([w_img, h_img])
    if ((points < 0) | (points > bounds)).any().item():
        raise ValueError("Accepted target outside original image bounds")
    uv = points / bounds * points.new_tensor([w, h])
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32) + 0.5,
        torch.arange(w, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    d2 = ((xx[None] - uv[:, 0, None, None]).square()
          + (yy[None] - uv[:, 1, None, None]).square())
    mask = (d2 <= radius_patches ** 2).any(dim=0)
    nearest = d2.flatten(1).argmin(dim=1)
    mask.view(-1)[nearest] = True
    if mask.all().item():
        raise ValueError("Every patch is positive; choose a different query/radius")
    return mask.flatten()

def marginal_positive_loss(logits, positive_mask):
    """Negative log probability assigned to ANY positive patch."""
    z = logits.float().reshape(-1)
    mask = positive_mask.reshape(-1).to(device=z.device, dtype=torch.bool)
    if mask.shape != z.shape or not mask.any().item() or mask.all().item():
        raise ValueError("Need an informative, nonempty positive mask")
    if not torch.isfinite(z).all().item():
        raise FloatingPointError("Non-finite similarity logits")
    return -torch.logsumexp(F.log_softmax(z, dim=0)[mask], dim=0)


def matching_loss(fs, ft, src_k, accepted_targets, target_wh, S, tau,
                  radius_patches = 1.0):
    """
    fs, ft: [C, H, W] feature maps
    src_k: [1, 2] source query in resized-image coordinates
    accepted_targets: [N, 2] targets in ORIGINAL target-image coordinates
    target_wh: ORIGINAL target-image (width, height)
    """
    q = sample_features(fs, src_k.to(fs.device), S)
    logits = (q @ ft.flatten(1)).reshape(-1) / tau

    positive_mask = positive_patch_mask(
        accepted_targets,
        target_wh,
        ft.shape[-2:],
        device= ft.device,
        radius_patches=radius_patches,
    )

    return marginal_positive_loss(logits, positive_mask)


def finetune(cfg, train_ds, val_ds, val_every_epoch = True):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = cfg.resolved_device()
    model = DinoFeatures(cfg).to(device)
    model.unfreeze_last(cfg.ft_unfreeze_blocks)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=cfg.ft_lr, weight_decay=0.05)
    loader = DataLoader(train_ds, batch_size=cfg.ft_batch_size, shuffle=True,
                        num_workers=cfg.num_workers, collate_fn=collate_keep, drop_last=True)
    steps = cfg.ft_epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.ft_lr, total_steps=steps, pct_start=0.05)
    use_amp = False
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
                matching_loss(
                    FS[b], 
                    FT[b], 
                    _to_resized(s["src_kps"], s["src_img"].size, S),
                    s["accepted_targets"],
                    s["trg_img"].size,
                    S,
                    cfg.ft_temperature,
                    radius_patches = 1.0,
                )
                for b, s in enumerate(batch)
            ) / len(batch)
            
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
                               model=model, device=device, ds = val_ds, verbose=False)
            score = summ["pck@0.1 (per-kp)"]
            history.append({"epoch": ep, **summ})
            print(f"epoch {ep}: val PCK@0.1 = {score:.2f}")
            if score > best:
                best = score
                torch.save(trainable_sd, out / "best.pth")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return best
