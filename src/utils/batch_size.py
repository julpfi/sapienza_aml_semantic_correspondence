"""Largest training micro-batch that fits the GPU (docs/PRD_finetuning.md section 5.3).

Memory depends on the model, image_size and the number of trained blocks, not on the data, so the
probe runs real forward+backward steps on random images of the training size: micro-batch 1, 2, 4, ...
up to effective_batch, until the GPU runs out of memory. The result is the largest size that worked,
times `safety` when a larger one failed (room for the optimiser state and fragmentation).
"""

import numpy as np
import torch
from PIL import Image


def fake_samples(n, cfg, rng):
    size = cfg["image_size"]
    samples = []
    for i in range(n):
        imgs = [Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)) for _ in range(2)]
        samples.append({"src_img": imgs[0], "trg_img": imgs[1], "src_image": f"probe{i}s", "trg_image": f"probe{i}t",
                        "src_xy": rng.uniform(20, size - 20, 2).astype(np.float32),
                        "trg_xys": rng.uniform(20, size - 20, (2, 2)).astype(np.float32)})
    return samples


def check_attention_kernel(run_step):
    """Warn when the fused SDPA kernels cannot run (the math kernel stores the full N x N attention)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            run_step(1)
        return True
    except RuntimeError as err:
        print(f"Warning: fused attention kernels unavailable ({str(err).splitlines()[0]}); "
              "SDPA falls back to the math kernel, which needs much more memory.")
        return False
    finally:
        torch.cuda.empty_cache()


def find_micro_batch(run_step, max_batch, device, safety=0.85):
    """Return (micro_batch, peak_gb). run_step(b) runs one forward+backward with b samples."""
    if torch.device(device).type != "cuda":
        return min(max_batch, 2), None  # CPU: small batches, only for checks
    check_attention_kernel(run_step)
    best, peak = 0, None
    for b in sorted({min(2 ** k, max_batch) for k in range(max_batch.bit_length() + 1)}):  # 1, 2, 4, ..., max
        try:
            torch.cuda.reset_peak_memory_stats()
            run_step(b)
            best, peak = b, torch.cuda.max_memory_allocated() / 2**30
        except torch.cuda.OutOfMemoryError:
            break
        finally:
            torch.cuda.empty_cache()
    if best == 0:
        raise RuntimeError("Even a micro-batch of 1 does not fit the GPU: lower image_size or unfreeze_blocks.")
    micro = best if best >= max_batch else max(1, int(best * safety))
    print(f"Batch probe: largest micro-batch that fits = {best} (peak {peak:.1f} GB); using {micro}.")
    return micro, peak
