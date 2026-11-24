# Literature review: Stable Diffusion + DINOv2 (incl. ViT-L) fusion for semantic correspondence (2026-11-24)

**Question:** Which recipe for fusing Stable Diffusion (SD) features with DINOv2 (in particular ViT-L/14) is best supported for SPair-71k style PCK@0.10 (bbox), so that we can implement it on a Colab T4 with about 517 training images?

**Short answer:** The best-supported recipe is SD+DINO (Zhang et al. 2023) for zero-shot and its supervised extension GeoAware-SC (Zhang et al. 2024), which keep both backbones frozen and train only a small bottleneck adapter on top of cached features. No paper I could verify reports DINOv2-L with SD, so ViT-L is an untested extrapolation. Our current single up-block, single-noise setup is a simplified DIFT-style variant of that recipe, and multi-layer SD features are the main missing piece.

## Sources
| # | Paper (authors, venue, year) | Link | What we use from it |
|---|---|---|---|
| 1 | A Tale of Two Features: SD complements DINO (Zhang, Jampani, Sun, Herrmann et al.), NeurIPS 2023 | https://arxiv.org/abs/2305.15347 | SD layers, PCA, alpha, resolution, zero-shot numbers |
| 2 | Telling Left from Right: Geometry-Aware Semantic Correspondence (Zhang et al.), CVPR 2024 | https://arxiv.org/abs/2311.17034 | Supervised adapter, loss, flip augmentation, window |
| 2b | GeoAware-SC code (`pck_train.py`, `projection_network.py`, `preprocess_map.py`) | https://github.com/Junyi42/GeoAware-SC | Exact defaults (read via raw files) |
| 3 | Emergent Correspondence from Image Diffusion, DIFT (Tang et al.), NeurIPS 2023 | https://arxiv.org/abs/2306.03881 | SD 2.1, up-block 1, t=261, ensemble 8, prompt |
| 4 | Diffusion Hyperfeatures (Luo et al.), NeurIPS 2023 | https://arxiv.org/abs/2305.14334 | Learned multi-layer, multi-timestep aggregation |
| 5 | Distillation of Diffusion Features for Semantic Correspondence, DistillDIFT (Fundel et al.), WACV 2025 | https://arxiv.org/abs/2412.03512 | DINOv2-B+reg with LoRA, re-measured SD/DINO numbers |
| 6 | CleanDIFT: Diffusion Features without Noise (Stracke et al.), CVPR 2025 | https://arxiv.org/abs/2412.03439 | No-noise features (only partially read) |
| 7 | Jamais Vu: generalization gap in supervised semantic correspondence, 2025 | https://arxiv.org/abs/2506.08220 | Supervised overfits to seen keypoints |
| 8 | Do It Yourself (DIY-SC): pseudo-labels, 2025 | https://arxiv.org/abs/2506.05312 | Same frozen-backbone adapter; 71.6 per-image |

## Findings

### Q1. SD+DINO architecture [1] and GeoAware-SC code [2b]
- [1] REPORTED (Sec. 3.1, 3.4): SD v1-5; decoder layers 2, 5, 8 (decoder output only, not the skip-concatenated feature); PCA computed jointly on source and target features, per layer; PCA output 256 dims when fusing with DINOv2-B/14; upsample to a common grid; fusion F = (alpha * ||F_SD||, (1 - alpha) * ||F_DINO||) with every part L2-normalised first, alpha = 0.5 (chosen empirically, ablation in the supplement).
- [1] REPORTED: timestep t = 100 by default; no real prompt (implicit captioner); SD input 960 and DINOv2 input 840, both giving a 60x60 grid; DINOv2 token facet of layer 11 (Table 1: token L11 beats key L11 and token L9 for ViT-B/14).
- [2b] REPORTED (code): `preprocess_map.py` uses SD v1-5, 960 px, 50 scheduler steps, `block_indices=[2,5,8,11]`, `NUM_ENSEMBLE = 1` (single noise sample), DINOv2 `dinov2_vitb14` layer 11 token at 840 px. The `co_pca` utility defaults to 128 dims per SD layer. The exact t and captioner are inside ODISE's extractor and were not visible to me: UNSURE about the exact timestep in the code (the paper says 100, the paper for GeoAware says 50).
- [2] REPORTED (App. A): SD-1-5 decoder layers 2, 5, 8, timestep 50, implicit captioner. Paper [1] says t=100 and paper [2] says t=50: the two papers disagree, so treat t in the 50 to 100 range as a flat region (INFERRED).
- [3] REPORTED: DIFT_sd uses SD 2.1, up-block index 1, t = 261, features averaged over 8 noise samples (4 under memory limits), prompt "a photo of a [class]", 768 px input, 1/16 resolution, 1280 dims. Null prompt: per-point PCK 59.5 to 57.6. No ensemble: 59.5 to 57.9 (SPair-71k, PCK@0.10 bbox, per point).
- UNSURE: mapping from the "layer 2/5/8" numbering to diffusers `up_blocks[i]` indices. Layer numbers count the 12 resnet-level stages of the ODISE/LDM decoder (3 per up-block). INFERRED: layers 2, 5, 8 are the last stage of up_blocks[0], [1], [2]. The channel counts in [1] (1280+960+480) and in the GeoAware code (640, 1280, 1280) do not agree with each other, so verify with a shape print before implementing.

### Q2. Numbers (SPair-71k, PCK@0.10 bbox)
- [1] REPORTED (Table 3, per point, zero-shot, nearest neighbour): DINOv2-B/14 55.6; SD 57.2; Fuse-B/14 64.0. Supervised bottleneck on top: DINOv2-B 69.9; SD 66.3; Fuse 74.6.
- [1] REPORTED: only ViT-S/14 and ViT-B/14 are evaluated. Table 1 token L11: DINOv2-S 52.7, DINOv2-B 55.7. No ViT-L or ViT-g result exists in the paper; the supplement was not available to me.
- [5] REPORTED (Tab. 1, 7, 8, 9): DINOv2-B 840 px 59.5 (Tab. 1) or 57.69 (Tab. 8, the two tables disagree); SD 1.5 56.24 (768 px, t=201, layer 5); SD 2.1 57.87 (768 px, t=261, layer 8); DINOv2+SD1.5 64.0 at 960 px but 62.03 in a 840 px configuration (Tab. 9). Higher SD resolution matters: 960 px is better than 768 px in [5]. DistillDIFT (DINOv2-B+reg with LoRA rank 8, distilled from DINOv2+SDXL Turbo) 65.1 unsupervised; supervised fine-tune 80.19.
- [7] REPORTED: per-image average, PCK@0.10 bbox: DINOv2 55.3; DINOv2+SD 61.1; GeoAware-SC supervised 83.2 on seen keypoints but 56.9 on unseen keypoints (SPair-U).
- UNSURE: whether DINOv2-L with SD beats DINOv2-B with SD. I found no verified number for it. [5] and [8] do not report ViT-L; [7] does not state the size. All published fusion numbers use ViT-B/14. Our own repo measured the DINOv2-L upgrade for DINOv2 alone (see `UNIFIED_LORA_LARGE` in `src/config.py`), which is the only local evidence.

### Q3. GeoAware-SC supervised recipe [2], [2b]
- REPORTED (Fig. 8, App. A): both SD and DINOv2 are frozen, features pre-extracted offline, only a post-processor is trained. SD s3, s4, s5 (640, 1280, 1280 channels) and DINOv2 (768) are concatenated along channels (3968 channels) at a 60x60 grid, and the post-processor is an `AggregationNetwork` from Diffusion Hyperfeatures [4] (about 5M parameters): one bottleneck stage per input layer (1x1, 3x3, 1x1, group norm, bottleneck width projection_dim/4 = 192, output 768 with default `PROJ_DIM` 768; the paper text for the model says "four bottleneck layers"), outputs combined with softmax-weighted learned mixing weights, then L2-normalised. Feature-map dropout 0.2 (code default). Note: my fetch of the code reported 96/384 widths, which assumes projection_dim 384; the default in `pck_train.py` is 768. Widths are UNSURE.
- REPORTED: loss = L_dense + L_sparse. L_sparse is a CLIP-style symmetric contrastive loss on keypoint descriptors. L_dense is a soft-argmax L2 distance to the target keypoint, with Gaussian noise added to the ground-truth positions.
- REPORTED (Tab. 5, 8, 12): soft-argmax window 15x15 around the argmax at test time for supervised models, 11x11 for the unsupervised one; using the window during training hurt looser thresholds (`SOFT_TRAIN_WINDOW` default 0).
- REPORTED: pose-variant augmentation with horizontal flip of one image (single flip), of both (double flip), and self flip, sampled with weights 1, 1, 0.25, with left/right keypoint labels swapped on flipped images. Test-time flip alignment in the zero-shot setting.
- REPORTED: AdamW, weight decay 1e-3, one-cycle, peak lr 1.25e-3 (0.3 warm-up fraction), batch size 1, 100k steps on SPair-71k (about 2 epochs of 53k pairs), one RTX 3090. Inputs 960 (SD) and 840 (DINOv2).
- REPORTED numbers (PCK@0.10 bbox): zero-shot 65.4 per image, 69.6 per point (Tab. 2, 3); supervised 82.9; supervised with AP-10K pre-training 85.6.
- REPORTED [8]: the same frozen backbones plus adapter recipe (lr 5e-3, 200k steps) with pseudo-labels reaches 71.6 per image and 74.4 per point. No DINOv2-L ablation; no LoRA.

### Q4. Fine-tune DINOv2 with LoRA and frozen SD, or frozen both plus adapter?
- REPORTED: the SD+DINO and GeoAware-SC papers only use frozen backbones with a trained adapter. A bottleneck on frozen DINOv2-B adds about 14 points (55.6 to 69.9) and on the fusion about 10.6 points (64.0 to 74.6) [1, Tab. 3].
- REPORTED [5]: LoRA (rank 8) on DINOv2-B+reg is used in a distillation setting and gives 80.19 supervised on SPair-71k. This is a different dataset scale (full SPair train set, 53k pairs) from ours.
- REPORTED [7]: supervised training on seen keypoints drops sharply on unseen keypoints (83.2 to 56.9), i.e. the adapter and fine-tuned models memorise keypoint types. With 517 images this risk is larger, and the unsupervised DINOv2+SD baseline (59.4 on unseen) beats supervised Geo-SC there.
- INFERRED: no paper compares "LoRA on DINOv2 + frozen SD concat" against "frozen both + adapter" head to head. The evidence supports training a small adapter on frozen features as the default. LoRA on the last blocks (already in our `UNIFIED_LORA` preset, which scored 0.64851 on the leaderboard according to the config comment) is supported only indirectly by [5]. A cheap and defensible design is LoRA on DINOv2 plus a zero-init residual adapter on SD, which our `adapter` mode already implements.

### Q5. Hyperfeatures [4] (for multi-layer, multi-timestep aggregation)
- REPORTED: SD v1-5, all 12 decoder layers, T = 50 scheduler steps keep every 5th step (11 timesteps), shared bottleneck per layer and learned mixing weights; AdamW lr 1e-3, batch 2, up to 5,000 steps. SPair-71k PCK@0.10: image-size normalisation 72.56 (single SD layer 58.80), bbox normalisation 64.61 (single SD layer 46.58). The image-size numbers are not comparable to our bbox metric. Number of noise samples: single inversion chain, no ensemble.

## Mapping to this repo
| Idea | Paper value | Current BASE / module | Proposed change | Expected effect |
|---|---|---|---|---|
| SD version | v1-5 in [1], [2]; 2.1 in [3] | `sd_model_id` = sd2-community/stable-diffusion-2-1 | Keep SD 2.1: DIFT's best (57.87 vs 56.24 at 768 px in [5] Tab. 7) and the only version public to us | Neutral to small positive |
| Up-block(s) | 3 decoder layers (s3, s4, s5) [1], [2] | `sd_up_block` = 1 (single) | New code: allow a list of up-blocks (e.g. [0, 1, 2]), resize each to the DINO grid, concat | Fewer wrong_keypoint errors (semantic from low-res, spatial from high-res); needs new code |
| Timestep | 50 to 100 (SD1.5 scale) [1], [2]; 261 [3] | `sd_timestep` = 261 | Keep 261 for SD 2.1 (DIFT), test 100 as a single ablation | Likely flat |
| Prompt | implicit captioner [1]; "a photo of a class" [3]; null = -1.9 pt [3] | empty prompt | Optional: class-name prompt if the category is available in test data (confirm); otherwise keep empty | About +1.9 pt per point from the DIFT ablation, INFERRED transfer |
| Noise samples | 1 in the GeoAware code; 8 in DIFT | one noise sample | Ensemble of 4 noise samples at inference (cached for training); no ensemble costs 1.6 pt in DIFT [3] | Fewer near_miss errors; 4x SD cost at inference |
| SD resolution | 960 SD vs 840 DINO [1], [2]; 768 in DIFT | `sd_image_size` = 736 | Raise toward 960 on T4 for inference only (cache features for training if memory is limiting); DINOv2 at 840 gives a 60x60 grid, and SD 960 gives a 60x60 map at 1/16 | [5] Tab. 9: 960 px 64.0 vs 840 px configuration 62.03; fewer near_miss errors |
| DINOv2 input | 840 (60x60) [1] | `image_size` 644 (46x46) | Raise to 840 for the base config; T4 memory for ViT-L at 840 (3600 tokens) is the limit: check | Finer grid, fewer near_miss errors |
| DINOv2 layer | layer 11 token, last [1], [2] | last-layer tokens | Keep last layer; for ViT-L the equivalent is the last layer, not layer 11 (UNSURE: no ViT-L layer ablation) | Neutral |
| Fusion weight | alpha 0.5 after per-part L2 norm; PCA 256 for SD [1] | `sd_concat_weight` = 0.5 in `normalize(cat(dino, 0.5*sd))` | Our concat uses weight 0.5 on SD against 1.0 on DINO, i.e. a 1:2 ratio, whereas [1] uses 1:1 (alpha 0.5 on both). Try `sd_concat_weight` = 1.0 | Possible +; paper's SD is also stronger due to multi-layer PCA |
| PCA | 256 dims per fused SD, joint over the pair [1] | none | Optional new code; the adapter's 1x1 input convolution plays a similar role in the supervised setting | Low priority |
| Adapter | 4 bottleneck stages, about 5M parameters, mixing weights, dropout 0.2 [2b] | `adapter_layers` = 2, `adapter_dim` = 256, lr 3e-4 | Keep 2 layers; test `adapter_layers` = 4 and add feature dropout 0.2 | Regularisation matters with 517 images |
| Adapter lr | 1.25e-3 (batch 1, one-cycle) [2b] | `lr_adapter` = 3e-4 | Test 1e-3 | Faster convergence; risk of flattened maps noted in config |
| Soft-argmax window | 15 supervised, 11 zero-shot [2] | `soft_window` = 11 | Test 15 for supervised heads (not during training) | Fewer near_miss errors |
| Flip augmentation | single/double/self flip, weights 1, 1, 0.25, swap left/right labels [2] | check `docs/decisions.md` | Add if absent; central to left/right flips | Directly targets left/right flips |
| LoRA | rank 8 on DINOv2-B [5] | `lora_rank` 8, `lora_blocks` 4 (6 for large), `lr_lora` 1e-4 (8e-5) | Keep | Supported indirectly |

## Proposed experiments
1. `sd_concat_weight` 0.5 to 1.0, one change, two seeds (matches the 1:1 weight in [1]).
2. `fusion` = adapter with `adapter_layers` 2 versus 4, and with feature dropout 0.2 (new small code), two seeds.
3. SD noise ensemble of 4 at inference only (new code), then multi-up-block SD (up-blocks 0, 1, 2 resized to the DINO grid) as a separate run.
4. `sd_image_size` 736 to 960 with DINOv2 `image_size` 644 to 840 (check T4 memory with ViT-L; fall back to base).
5. DINOv2-L versus DINOv2-B under the same SD fusion (`unified_lora_large` plus `fusion`): this is the untested comparison and should be run once on both backbones with two seeds; it is the only way to know whether L helps with SD.
6. `soft_window` 11 versus 15 on the best supervised head.

## Open questions / unsure
- No verified DINOv2-L plus SD number exists; the supplements of [1] and [2] were not read (the supplement hosts the alpha ablation and the timestep ablation).
- Timestep disagreement between [1] (100) and [2] (50); the code timestep sits inside ODISE.
- Adapter widths in `projection_network.py` depend on `PROJ_DIM` (default 768); the fetched summary of the code gave 96/384, so the exact widths are not confirmed.
- The "layer 2/5/8" to diffusers up-block mapping is inferred, not verified.
- CleanDIFT [6] was only read through search snippets: it reports +1.2 PCK over DIFT SD 2.1 with 8x ensembling and no prompt, but its checkpoints need a new dependency or weights, so it is out of scope unless approved.
- Whether our test data exposes the category (for a "a photo of a class" prompt) is not checked.
