# SOCO: a DINO-first semantic correspondence pipeline

The task: given a keypoint on an object in a source image, find the same semantic point (for example, a cat's left ear) on another object of that category in a target image. Results are scored with PCK@α: a prediction counts as correct when it lands within α·max(bbox w, h) of the ground truth.

The pipeline starts with the cheapest strong baseline, frozen DINOv2 features. It moves to heavier methods only when validation results show they are needed.

```
Stage 1  frozen DINOv2 + matching ──► decide ──► target met? ──► submit
                                         │ no
Stage 2  fine-tune last blocks ───────► decide ──► target met? ──► submit
                                         │ no
Stage 3  advanced: LoRA / higher res, geometry-aware matching, SD+DINO fusion
```

## Layout

| File | Role |
|---|---|
| `soco/config.py` | One `Config` dataclass per experiment; saved next to every result |
| `soco/data.py` | `SPairDataset` (SPair-71k format) and `CSVPairDataset` (generic table). Add an adapter here if the competition format differs |
| `soco/backbone.py` | DINO patch features via torch.hub (DINOv2/DINOv3 repo, offline-capable) or Hugging Face |
| `soco/matching.py` | Keypoint transfer: bilinear feature sampling, cosine similarity map, argmax or windowed soft-argmax, optional flip test |
| `soco/metrics.py` | PCK@α per keypoint, per image, per category |
| `soco/evaluate.py` | Stage 1 evaluation, cached matching sweep, feature sweep |
| `soco/decide.py` | Decision gate: reads validation results and prints the next step |
| `soco/finetune.py` | Stage 2: unfreeze last N blocks, soft-label matching loss, AMP, best-checkpoint by val PCK |
| `soco/submit.py` | Writes `id,prediction` rows (`"[x,y]"` in original target pixels), aligned and checked against `data/sample_submission.csv` |
| `data/sample_submission.csv` | The competition's sample submission (577 query ids) |
| `run.py` | CLI for all of the above |
| `soco_kaggle.ipynb` | The same flow as a Kaggle notebook |

## Commands

```bash
python run.py eval      --set data_root=/path/SPair-71k max_pairs=300
python run.py sweep     --set max_pairs=300          # matcher / upsample / temperature / flip (one feature pass)
python run.py featsweep --set max_pairs=300          # layer x img_size
python run.py decide    --run outputs/stage1_dinov2_b14 --target 70
python run.py finetune  --set run_name=stage2_ft2 ft_unfreeze_blocks=2 ft_epochs=2
python run.py decide    --run outputs/stage2_ft2/val_ep1 --baseline outputs/stage1_dinov2_b14 --stage 2
python run.py predict   --set data_format=csv csv_path=<test queries csv> image_dir=<images dir> \
                        ft_checkpoint=outputs/stage2_ft2/best.pth \
                        --sample_submission data/sample_submission.csv --out submission.csv
```

## Verified results (SPair-71k val, 150 random pairs, CPU, DINOv2 ViT-S/14 at 448 px)

These numbers come from smoke tests, not tuned runs; ViT-B/14 at 518 px on a GPU should do better.

| Configuration | PCK@0.05 | PCK@0.1 | PCK@0.15 |
|---|---|---|---|
| argmax, no upsampling | 33.5 | 51.5 | 60.9 |
| soft-argmax window, 2× feature upsampling | 38.9 | 54.3 | 62.3 |
| same + naive flip test | 34.4 | 47.7 | 56.1 |

What these results show:
- Windowed soft-argmax plus feature upsampling is the default because it gave the best score (+2.8 PCK@0.1, +5.4 PCK@0.05).
- The naive flip test hurts. DINO features are close to mirror-symmetric, so the global alignment score picks the flipped source about 57% of the time. It stays off by default. Proper pose-aware alignment belongs in Stage 3.
- The weakest categories are pottedplant (16.9), car, tvmonitor and boat (all about 38). This matches the literature: texture-poor and symmetric objects are where SD+DINO fusion and geometry-aware methods help most.
- The fine-tuning loop, checkpoint reload, Hugging Face backbone path and submission writer were each run end to end on small subsets.

## Submission format (from the competition's sample_submission.csv)

```
id,prediction
test_q_00341,"[250.0,167.5]"
```

- There is one row per query keypoint, 577 rows in total. `prediction` is `[x, y]` in original target-image pixels.
- The sample's placeholder values are image centres (e.g. 250×187.5 for a 500×375 image). That confirms the coordinates are in the original image size, not a resized one.
- The writer reorders rows to match the sample. It warns about missing or extra ids, and any missing id falls back to the sample's placeholder.
- Test queries are read with `data_format="csv"`, one row per query. Columns: `pair_id` (the query id), `src_image`, `trg_image` and the source point, given either as `src_x`/`src_y` or as `src_kps="[x,y]"`. If the competition's test file uses different column names, rename them or pass a `colmap`.
- Features are cached per image, so queries that share an image pair reuse them. On the 577-query mock test this ran about 15× faster on CPU (50 s vs about 12 min).
- End-to-end test: a mock test file mapped the 577 sample ids to SPair-71k validation keypoints. The output matched the sample's ids and order, every prediction parsed as `[x, y]`, and PCK@0.1 was 58.2 (ViT-S/14, 448 px).

## Running on Kaggle with internet disabled

1. In a notebook with internet on, run `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")` and save `model.state_dict()` to a `.pth` file. Also copy the DINOv2 repo source.
2. Upload this repo and the DINOv2 assets as Kaggle datasets, then attach them.
3. Set `hub_repo=/kaggle/input/.../dinov2`, `hub_local=True` and `weights_path=/kaggle/input/.../dinov2_vitb14.pth`.

## Before relying on it, check

- Competition rules: are external pretrained weights allowed, and under which licences? DINOv2 is Apache-2.0. DINOv3 weights are gated under Meta's DINOv3 licence.
- Data format: if the data is not in SPair-71k layout, use `data_format="csv"` with a column mapping, or add an adapter.
- The metric's normalisation (bbox vs whole image) and α. Set `pck_norm` and `pck_alphas` to match.
- The test-query file's column names: the submission schema itself is already handled.
