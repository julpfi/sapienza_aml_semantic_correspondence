# SD + DINOv2-large: trained parameters and architecture

Preset `unified_lora_large_sd` in `src/config.py` (decision D-009 in [decisions.md](decisions.md)). Run book:
[RUN.md](RUN.md), section "SD + DINOv2-large (D-009)". Literature basis:
[literature/2026-11-24-sd-dinov2-large-fusion.md](literature/2026-11-24-sd-dinov2-large-fusion.md).

## The layer that merges SD and DINOv2

The merge layer is `Fusion` in `src/models/fusion.py`, mode `fusion: "adapter"`. It is on in the preset
`unified_lora_large_sd` and off in `BASE` (`fusion: "none"`, D-010).

    fused = L2normalize( dino + out( blocks( inp( Dropout2d(sd) ) ) ) )

- `dino` is the DINOv2-L feature map (1024 channels, already L2-normalised per cell). There is no separate DINOv2
  head: `head` is "none", so the head is the identity and the LoRA-adapted backbone output goes straight into the
  merge.
- The SD branch is projected to 1024 channels and added as a residual. The last convolution (`out`) starts at
  zero, so at update 0 `fused == dino`: training starts from the DINOv2-L LoRA model, and the SD part only adds
  what the loss finds useful.
- `cross_transform` (a second, cross-image adapter after the merge) exists but is off in the preset.

## The learned merge: `fusion: "adapter_mix"` (D-011, off by default)

Preset `unified_lora_large_sd_mix` = `unified_lora_large_sd` with `fusion` "adapter_mix". `BASE` keeps
`fusion: "none"`. Run it only after `unified_lora_large_sd` (the baseline with the fixed sum), with the same seeds.

    s     = out( blocks( inp( Dropout2d(sd) ) ) )          # the same SD adapter as in "adapter", out starts at 0
    fused = L2normalize( mix( cat(dino, s) ) )             # mix: 1x1 conv, 2048 -> 1024 channels, + bias

- **What changes:** in "adapter" the two branches are combined by a fixed sum, `dino + s`. In "adapter_mix" the
  sum is replaced by a learned 1x1 convolution over the concatenation `[dino ; s]` (2048 channels per cell).
  Written per cell: `mix(cat(dino, s)) = W_d · dino + W_s · s + b`, with `W_d` and `W_s` two learned 1024 x 1024
  matrices. Every output channel can therefore take any combination of DINOv2-L channels and SD channels, and
  the balance between the two branches settles during fine-tuning. The L2 normalisation stays after the merge,
  because matching uses cosine similarity.
- **Initialisation:** `W_d = I`, `W_s = I`, `b = 0`. At update 0, `mix(cat(dino, s)) = dino + s`, which is exactly
  the "adapter" mode, and since `out` starts at zero, exactly `dino`. The run starts from the same model as the
  baseline, so a paired comparison measures only the learned merge.
- **Why not `W_s = 0`:** with `out` at zero, `W_s = 0` would give the whole SD branch a zero gradient forever.
  With `W_s = I`, `out` gets a gradient at the first step; the `W_s` half starts moving from the second step,
  once `s` is no longer zero.
- **Training:** `mix` belongs to the adapter parameter group (`lr_adapter` 3e-4, weight decay 0.05) and is saved
  in `best.pt` as `fusion.mix.weight` / `fusion.mix.bias`. A smaller lr for `mix` alone is a possible tuning step;
  it is not a separate key today.

## Architecture

```
 source / target image (original pixels)
        |
        |  preprocess: resize to a 644 x 644 square, ImageNet normalisation
        v
 +------+-----------------------------------------------+
 |                                                      |
 v                                                      v
 DINOv2-L/14 (facebook/dinov2-large, 24 blocks, 1024-d) Stable Diffusion 2.1 (frozen, fp16 on GPU, no grad)
  blocks 0-17   frozen                                  resize to 736 x 736 -> VAE encoder -> latent 92 x 92
  blocks 18-23  frozen weights + LoRA (r 8, alpha 16)    + noise at t = 261 (fixed seed), empty prompt
                on q, k, v, o, fc1, fc2                  -> UNet decoder, forward stopped after up_block 2
                + trained LayerNorms (norm1, norm2)        up_block 0: 1280 x 23 x 23
  final LayerNorm  trained                                 up_block 1: 1280 x 46 x 46
        |                                                  up_block 2:  640 x 92 x 92
        |  patch tokens -> 1024 x 46 x 46                  each resized to 46 x 46, L2-normalised,
        |  L2-normalised per cell                          concatenated, L2-normalised -> 3200 x 46 x 46
        |                                                  (evaluation: average of 4 noise samples)
        |  head "none" (identity)                                  |
        |                                                          v
        |                                             Fusion adapter (trained)
        |                                               Dropout2d 0.2 (training only)
        |                                               inp: 1x1 conv 3200 -> 1024
        |                                               2 x residual bottleneck:
        |                                                 1x1 1024->256, GELU, 3x3 256->256, GELU, 1x1 256->1024
        |                                               out: 1x1 conv 1024 -> 1024 (zero init)
        |                                                          |
        +---------------------------> MERGE <----------------------+
                                       |
                 "adapter" (baseline): dino + s      (fixed sum, no parameters)
                 "adapter_mix":        mix(cat(dino, s))
                                       1x1 conv 2048 -> 1024 + bias, init [I | I], 0
                                       = W_d·dino + W_s·s + b, learned per channel
                                       |
                                       v
                         L2-normalise -> fused features 1024 x 46 x 46
                                       |
                 +---------------------+-----------------------------+
                 |                                                   |
         training                                             inference
   target map upsampled x2 (92 x 92)              cosine similarity of the source-point
   softmax over target cells, T 0.06 -> 0.03      descriptor with the target map (upsampled x2),
   marginal loss on cells near any accepted       soft-argmax in an 11 x 11 window (T 0.02),
   target (adaptive radius) + hard negatives      2-stage object zoom (stage 2 re-runs DINOv2
   (weight 5) from the pair's other queries       and SD on the crops), mapped back to
                                                  original target-image pixels
```

## Trained parameters

Counts for DINOv2-L (hidden 1024, MLP 4096) and SD 2.1 with `sd_up_blocks` [0, 1, 2]. They were computed with the
project's own `LoRALinear` and `Fusion` classes; the same formulas give the 3.32M trainable parameters that the CPU
smoke run with DINOv2-base printed.

| Part | Module | What is trained | Parameters | Peak lr (OneCycleLR) | Init |
|---|---|---|---:|---|---|
| DINOv2-L blocks 18-23 | `LoRALinear` on `q_proj`, `k_proj`, `v_proj`, `o_proj` | `lora_A` [8, 1024], `lora_B` [1024, 8]: 16,384 per layer, 4 layers x 6 blocks | 393,216 | `lr_lora` 8e-5 | A Kaiming, B zero |
| DINOv2-L blocks 18-23 | `LoRALinear` on `fc1` (1024 -> 4096), `fc2` (4096 -> 1024) | 40,960 per layer, 2 layers x 6 blocks | 491,520 | `lr_lora` 8e-5 | A Kaiming, B zero |
| DINOv2-L blocks 18-23 | `norm1`, `norm2` LayerNorms (`lora_train_layernorm` True) | weight + bias, 2,048 each, 2 x 6 | 24,576 | `lr_lora` 8e-5 | pretrained |
| DINOv2-L | final `layernorm` | weight + bias | 2,048 | `lr_lora` 8e-5 | pretrained |
| **LoRA group** | | | **911,360** | | |
| Fusion adapter | `inp` 1x1 conv 3200 -> 1024 | weight + bias | 3,277,824 | `lr_adapter` 3e-4 | PyTorch default |
| Fusion adapter | `blocks`: 2 x bottleneck (1x1 1024->256, 3x3 256->256, 1x1 256->1024) | 1,115,648 per block | 2,231,296 | `lr_adapter` 3e-4 | PyTorch default |
| Fusion adapter | `out` 1x1 conv 1024 -> 1024 | weight + bias | 1,049,600 | `lr_adapter` 3e-4 | zero |
| **Adapter group** | | | **6,558,720** | | |
| **Total trained (`unified_lora_large_sd`)** | | | **7,470,080** | | |
| Learned merge (`adapter_mix` only) | `mix` 1x1 conv 2048 -> 1024 | weight 1024 x 2048 + bias 1024 | 2,098,176 | `lr_adapter` 3e-4 | weight `[I \| I]`, bias 0 |
| **Total trained (`unified_lora_large_sd_mix`)** | | | **9,568,256** | | |

Frozen (never trained, never saved in `best.pt`): DINOv2-L base weights (about 304M), SD 2.1 UNet (about 866M) and
VAE (about 84M). The SD text encoder is used once for the empty-prompt embedding and then deleted.

## Training settings of the preset

| Setting | Value | Key(s) |
|---|---|---|
| Epochs / warmup | 6 / 1.0 epoch, OneCycleLR, AdamW | `epochs`, `warmup_epochs` |
| Weight decay / gradient clip | 0.05 / 10.0 | `weight_decay`, `grad_clip` |
| Batching | one item = one image pair with all its queries, 4 pairs per update | `pair_batch`, `pair_effective_batch` |
| Loss | marginal, temperature 0.06 -> 0.03 over 2 epochs, loss on the x2 upsampled target map, adaptive positive radius | `loss`, `loss_temperature_start`, `loss_temperature`, `loss_temperature_epochs`, `loss_upsample`, `adaptive_pos_radius` |
| Hard negatives | weight 5.0 | `hard_neg_weight` |
| EMA | 0.99 on every trained parameter; validation and `best.pt` use the averaged weights | `ema_decay` |
| Dropout | LoRA 0.05, SD input of the adapter 0.2 | `lora_dropout`, `adapter_dropout` |
| Augmentation | pair flip 0.5, rotation 25 degrees, crop scale [0.45, 1.0], label jitter 0.25 patches | `flip_augment_prob`, `aug_rotation_deg`, `aug_crop_scale`, `label_jitter` |
| SD features | SD 2.1, 736 px, t 261, up-blocks [0, 1, 2], 1 noise sample in training, 4 at evaluation | `sd_model_id`, `sd_image_size`, `sd_timestep`, `sd_up_blocks`, `sd_noise_samples` |
| Matching | soft-argmax window 11, temperature 0.02, features upsampled x2, object zoom on | `soft_window`, `soft_temperature`, `feat_upsample`, `object_zoom` |

Not measured yet: validation PCK@0.10 of this preset. Compare it with `unified_lora_large` (seeds 0 and 1) using
`scripts/compare_runs.py`.
