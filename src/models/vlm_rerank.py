"""Zero-shot (or QLoRA-tuned) VLM Set-of-Mark Top-K peak re-ranking for Part 1 inference.

Two-phase VRAM-safe design (fits comfortably on a 15 GB Colab T4):
1. Phase 1 (DINOv2 / Ensemble): extract Top-K spatially separated NMS peaks from the final
   similarity map for every query.
2. Phase 2 (VLM): free DINOv2/Ensemble from GPU, load 4-bit Qwen2.5-VL-7B-Instruct, and
   score ONLY ambiguous queries where `Peak1 - Peak2 < effective_gap`.
   Because `score(k) = cos(k) + alpha * P_vlm(k)` and `0 <= P_vlm(k) <= 1`, Peak #2 can
   mathematically never overtake Peak #1 when `Peak1 - Peak2 >= alpha`, so gating on
   `effective_gap = min(vlm_gap, alpha)` is exact and skips ~70-80% of queries.
"""

import gc
import importlib.util
import subprocess
import sys

import numpy as np
import torch
from PIL import ImageDraw
from tqdm.auto import tqdm

from data.transforms import to_original
from src.models.matching import locate

# Distinct neutral marker colors (avoid RED on Image 2 so Marker #1 doesn't color-match the RED query circle on Image 1)
PALETTE = ("#00e5ff", "#f1c40f", "#2ecc71")


def build_vlm_prompt(top_k):
    """Set-of-Mark multiple-choice prompt for `top_k` candidate peaks (2 or 3)."""
    if top_k == 2:
        return (
            "Image 1 shows an object with a RED circle marking one specific semantic part. "
            "Image 2 shows another instance of the same object category (possibly rotated, flipped, "
            "or from a different 3D viewpoint) with numbered candidate points 1 and 2. "
            "Which candidate number (1 or 2) corresponds to the EXACT same semantic part as the red circle in Image 1? "
            "Pay close attention to 3D orientation, left vs. right side, and front vs. back. "
            "Reply with ONLY the single digit (1 or 2)."
        )
    nums = ", ".join(str(i) for i in range(1, top_k + 1))
    return (
        "Image 1 shows an object with a RED circle marking one specific semantic part. "
        "Image 2 shows another instance of the same object category (possibly rotated, flipped, "
        f"or from a different 3D viewpoint) with numbered candidate points {nums}. "
        f"Which candidate number ({nums}) corresponds to the EXACT same semantic part as the red circle in Image 1? "
        "Pay close attention to 3D orientation, left vs. right side, and front vs. back. "
        f"Reply with ONLY the single digit ({nums})."
    )


def extract_topk_peaks(sim_1hw, out_wh, offset_xy, cfg, k=2, nms_radius=10, sbox=None, tbox=None):
    """Extract top-K spatially separated local peaks from a [1, H, W] similarity map.

    Rank 1 uses the exact standard `locate(sim_1hw, ...)` so non-reranked queries are
    bit-for-bit identical to the baseline model.
    Returns a list of dicts: `[{"rank": 1..k, "score": float, "xy_orig": np.ndarray[2], "sbox": ..., "tbox": ...}]`.
    """
    size = int(cfg["image_size"])
    offset = np.asarray(offset_xy, dtype=np.float32)
    s = sim_1hw[0].clone()
    h, w = s.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=s.device), torch.arange(w, device=s.device), indexing="ij"
    )
    peaks = []
    for rank in range(1, int(k) + 1):
        flat = int(s.argmax().item())
        r, c = flat // w, flat % w
        val = float(s[r, c].item())
        if not np.isfinite(val):
            break
        score = float(sim_1hw[0, r, c].item())
        local_mask = ((yy - r) ** 2 + (xx - c) ** 2) <= (int(nms_radius) ** 2)
        if rank == 1:
            xy_res = locate(sim_1hw, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])[0]
        else:
            s_local = sim_1hw[0].masked_fill(~local_mask, float("-inf"))[None]
            xy_res = locate(s_local, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])[0]
        xy_orig = to_original(xy_res.cpu(), out_wh, size).numpy()[0] + offset
        peaks.append({
            "rank": rank,
            "score": score,
            "xy_orig": xy_orig.astype(np.float32),
            "sbox": sbox,
            "tbox": tbox,
        })
        s = s.masked_fill(local_mask, float("-inf"))
    return peaks


def draw_som_pair(src_img, trg_img, src_xy, peaks, max_side=448):
    """Draw Set-of-Mark circles on zoomed Stage-2 crops (when sbox/tbox are present) and cap resolution for T4 VRAM."""
    sbox = peaks[0].get("sbox") if peaks else None
    tbox = peaks[0].get("tbox") if peaks else None

    src_rgb = src_img.convert("RGB")
    trg_rgb = trg_img.convert("RGB")
    if sbox is not None:
        src = src_rgb.crop(sbox)
        sx, sy = float(src_xy[0]) - float(sbox[0]), float(src_xy[1]) - float(sbox[1])
    else:
        src = src_rgb.copy()
        sx, sy = float(src_xy[0]), float(src_xy[1])

    if tbox is not None:
        trg = trg_rgb.crop(tbox)
        tx_off, ty_off = float(tbox[0]), float(tbox[1])
    else:
        trg = trg_rgb.copy()
        tx_off, ty_off = 0.0, 0.0

    r_s = max(6, int(0.025 * max(src.size)))
    r_t = max(8, int(0.032 * max(trg.size)))
    ds = ImageDraw.Draw(src)
    ds.ellipse((sx - r_s, sy - r_s, sx + r_s, sy + r_s), fill="red", outline="white", width=3)

    dt = ImageDraw.Draw(trg)
    for idx, pk in enumerate(peaks):
        col = PALETTE[idx % len(PALETTE)]
        tx, ty = float(pk["xy_orig"][0]) - tx_off, float(pk["xy_orig"][1]) - ty_off
        dt.ellipse((tx - r_t, ty - r_t, tx + r_t, ty + r_t), fill=col, outline="black", width=2)
        dt.text((tx - r_t * 0.3, ty - r_t * 0.55), str(idx + 1), fill="black")

    if max_side and int(max_side) > 0:
        cap = (int(max_side), int(max_side))
        src.thumbnail(cap)
        trg.thumbnail(cap)
    return src, trg


def ensure_vlm_deps():
    """Install lightweight VLM quantization/helper packages on demand (e.g. fresh Colab runtime)."""
    missing = [
        pkg for pkg, mod in (
            ("bitsandbytes", "bitsandbytes"),
            ("accelerate", "accelerate"),
            ("qwen-vl-utils", "qwen_vl_utils"),
        )
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        print(f"[VLM rerank] Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def _score_one_som(vlm, proc, src_som, trg_som, prompt, tok_ids):
    """Single forward pass -> softmax probabilities over candidate digit tokens [K]."""
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": src_som},
            {"type": "image", "image": trg_som},
            {"type": "text", "text": prompt},
        ],
    }]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[src_som, trg_som], return_tensors="pt").to(vlm.device)
    logits = vlm(**inputs).logits[0, -1, tok_ids].float()
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    del inputs, logits
    return probs


def _load_vlm(cfg):
    ensure_vlm_deps()
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    vlm_id = cfg.get("vlm_model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_id, quantization_config=qcfg, device_map="auto"
    )
    lora_path = cfg.get("vlm_lora_path")
    if lora_path:
        from peft import PeftModel

        print(f"[VLM rerank] Loading QLoRA adapter from {lora_path}")
        vlm = PeftModel.from_pretrained(vlm, lora_path)
    vlm.eval()
    proc = AutoProcessor.from_pretrained(vlm_id)
    return vlm, proc, vlm_id


def rerank_predictions(dataset, pred_xys, peaks_list, cfg, model=None, cache=None, verbose=True):
    """Phase 2: offload correspondence model, load 4-bit VLM, and softly re-rank ambiguous peaks.

    Args:
        dataset: indexable dataset returning sample dicts (`src_img`, `trg_img`, `src_xy`, `query_id`).
        pred_xys: list of baseline `[2]` numpy arrays (one per query).
        peaks_list: list of Top-K peak lists from `predict_candidates()` (one per query).
        cfg: run config dict.
        model: correspondence model (temporarily offloaded to CPU to free T4 VRAM).
        cache: FeatureCache (cleared before loading VLM to free GPU/RAM).
    Returns:
        (updated_pred_xys, stats_dict)
    """
    top_k = max(2, int(cfg.get("vlm_top_k", 2)))
    alpha = float(cfg.get("vlm_alpha", 0.020))
    cfg_gap = cfg.get("vlm_gap")
    # Exact math: since 0 <= P_vlm(k) <= 1, Peak #2 can never overtake Peak #1 if gap >= alpha.
    effective_gap = float(alpha) if cfg_gap is None else min(float(cfg_gap), float(alpha))
    max_side = int(cfg.get("vlm_max_image_side", 448))
    swap_debias = bool(cfg.get("vlm_swap_debias", False))

    ambig_indices = []
    for i, peaks in enumerate(peaks_list):
        if len(peaks) >= 2 and (peaks[0]["score"] - peaks[1]["score"]) < effective_gap:
            ambig_indices.append(i)

    stats = {
        "vlm_rerank": True,
        "vlm_model_id": cfg.get("vlm_model_id", "Qwen/Qwen2.5-VL-7B-Instruct"),
        "vlm_top_k": top_k,
        "vlm_alpha": alpha,
        "vlm_gap": effective_gap,
        "vlm_swap_debias": swap_debias,
        "vlm_scored": len(ambig_indices),
        "vlm_flipped": 0,
    }
    if not ambig_indices or alpha <= 0.0:
        if verbose:
            print(f"[VLM rerank] 0 / {len(pred_xys)} queries have gap < {effective_gap:.4f}; skipping VLM load.")
        return list(pred_xys), stats

    # Free GPU memory occupied by DINOv2 / Ensemble and feature cache before loading 4-bit 7B VLM
    if cache is not None and hasattr(cache, "_store"):
        cache._store.clear()
    orig_device = None
    if model is not None:
        try:
            orig_device = next(model.parameters()).device
        except StopIteration:
            orig_device = None
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(
            f"[VLM rerank] Scoring {len(ambig_indices)} / {len(pred_xys)} ambiguous queries "
            f"(gap < {effective_gap:.4f}, top{top_k}, alpha={alpha:.4f}, swap_debias={swap_debias})..."
        )

    vlm, proc, _ = _load_vlm(cfg)
    prompt = build_vlm_prompt(top_k)
    tok_ids = [proc.tokenizer.encode(str(k), add_special_tokens=False)[-1] for k in range(1, top_k + 1)]

    updated = list(pred_xys)
    n_flipped = 0
    prob_sum = np.zeros(top_k, dtype=np.float64)
    with torch.no_grad():
        for i in tqdm(ambig_indices, disable=not verbose, desc="VLM rerank"):
            sample = dataset[i]
            cand = peaks_list[i][:top_k]
            k_cur = len(cand)
            cur_prompt = prompt if k_cur == top_k else build_vlm_prompt(k_cur)
            cur_toks = tok_ids[:k_cur]

            src_som, trg_som = draw_som_pair(
                sample["src_img"], sample["trg_img"], sample["src_xy"], cand, max_side=max_side
            )
            probs = _score_one_som(vlm, proc, src_som, trg_som, cur_prompt, cur_toks)

            if swap_debias and k_cur == 2:
                src_sw, trg_sw = draw_som_pair(
                    sample["src_img"], sample["trg_img"], sample["src_xy"], [cand[1], cand[0]], max_side=max_side
                )
                probs_sw = _score_one_som(vlm, proc, src_sw, trg_sw, cur_prompt, cur_toks)
                probs = 0.5 * (probs + probs_sw[::-1])

            if k_cur == top_k:
                prob_sum += probs
            scores = np.array([p["score"] for p in cand], dtype=np.float64) + alpha * probs
            best_idx = int(np.argmax(scores))
            if best_idx != 0:
                updated[i] = cand[best_idx]["xy_orig"]
                n_flipped += 1

    del vlm, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if model is not None and orig_device is not None and orig_device.type != "cpu":
        model.to(orig_device)

    mean_probs = (prob_sum / max(len(ambig_indices), 1)).round(3).tolist()
    stats["vlm_flipped"] = n_flipped
    stats["vlm_mean_probs"] = mean_probs
    if verbose:
        print(
            f"[VLM rerank] Done: scored {len(ambig_indices)}/{len(pred_xys)} ambiguous queries, "
            f"flipped {n_flipped} predictions (mean P_vlm={mean_probs})."
        )
    return updated, stats
