# Graph Report - advance  (2026-11-24)

## Corpus Check
- Corpus is ~34,791 words - fits in a single context window. You may not need a graph.

## Summary
- 623 nodes · 1486 edges · 20 communities (13 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Dataset Loading
- Coordinate Transforms
- Run Entry & CLI Config
- Imports & Test Suite
- Docs: Data & Decisions
- Pair Batch Checks
- DINOv2 Backbone
- Flip Augmentation
- LoRA & EMA Checks
- Run Comparison Bootstrap
- Ensemble Model
- Micro-batch Probe
- Cross-Image Transform
- On-disk Feature Store
- Diagnostics Plots
- Conda Environment
- Fusion Adapter Bottleneck

## God Nodes (most connected - your core abstractions)
1. `train()` - 43 edges
2. `RUN.md run guide` - 28 edges
3. `main()` - 23 edges
4. `get_config()` - 21 edges
5. `main()` - 19 edges
6. `build_dataset()` - 17 edges
7. `CorrespondenceModel` - 16 edges
8. `predict_with_zoom()` - 16 edges
9. `Plan: orientation-aware training + LoRA/EMA + SD fusion on branch sd` - 16 edges
10. `read_queries()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `train()` --uses--> `QueryDataset`  [INFERRED]
  src/train.py → data/datasets.py
- `train()` --indirect_call--> `collate_samples()`  [INFERRED]
  src/train.py → data/datasets.py
- `predict_with_zoom()` --calls--> `load_pair_source_xy()`  [EXTRACTED]
  src/models/zoom.py → data/datasets.py
- `check_pair_dataset()` --calls--> `QueryDataset`  [EXTRACTED]
  scripts/check_pair_batch.py → data/datasets.py
- `check_augment_pair()` --calls--> `read_queries()`  [EXTRACTED]
  scripts/check_pair_batch.py → data/datasets.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **v3 additions targeting near-miss / wrong-keypoint errors** — docs_plan_pair_batch, docs_plan_hard_negatives, docs_plan_lora, docs_plan_ema, docs_plan_sd_features, docs_plan_fusion_adapter [EXTRACTED 1.00]
- **Inference matching pipeline** — docs_run_dinov2_backbone, docs_run_concat_fusion, docs_run_soft_argmax, docs_run_pipeline [EXTRACTED 1.00]

## Communities (20 total, 7 thin omitted)

### Community 0 - "Dataset Loading"
Cohesion: 0.05
Nodes (25): build_dataset(), load_pair_source_xy(), _open_rgb(), QueryDataset, read_queries(), disjoint_folds_path(), load_disjoint_folds(), load_val_pair_ids() (+17 more)

### Community 1 - "Coordinate Transforms"
Cohesion: 0.06
Nodes (26): preprocess_image(), to_original(), to_resized(), _sinusoidal_2d(), alignment_score(), cell_centers(), choose_source(), compute_similarity_map() (+18 more)

### Community 2 - "Run Entry & CLI Config"
Cohesion: 0.05
Nodes (36): _apply_cli_overrides(), load_config(), load_trained_weights(), main(), parse_args(), preflight(), check(), head_model_class() (+28 more)

### Community 3 - "Imports & Test Suite"
Cohesion: 0.06
Nodes (24): install(), load_colab_secrets(), main(), mount_drive(), pip_requirements(), pull(), run(), version_check() (+16 more)

### Community 4 - "Docs: Data & Decisions"
Cohesion: 0.06
Nodes (49): Data README, csv_columns mapping, data_dir runtime resolution (local/Colab/Kaggle), Lang notebook, QueryDataset (one CSV row = one query), scale = object size L (longest bbox side), target_points (all accepted targets), transforms.py coordinate mapping (original <-> resized, flips) (+41 more)

### Community 5 - "Pair Batch Checks"
Cohesion: 0.07
Nodes (28): collate_samples(), base_cfg(), cell_xy(), check_augment_pair(), check_hard_negatives(), check_matching_loss_unchanged(), check_pair_batch_loss(), check_pair_dataset() (+20 more)

### Community 6 - "DINOv2 Backbone"
Cohesion: 0.06
Nodes (6): DinoBackbone, _hub_interpolate_pos_encoding(), ProjectionHead, CorrespondenceModel, FeatureCache, SDFeatures

### Community 7 - "Flip Augmentation"
Cohesion: 0.11
Nodes (19): clip_point(), flip_points(), flip_sample(), main(), run(), _augment_image(), augment_pair(), augment_sample() (+11 more)

### Community 8 - "LoRA & EMA Checks"
Cohesion: 0.10
Nodes (11): check(), main(), make_cfg(), raises(), randomise_B(), wrap_once(), apply_lora(), lora_parameters() (+3 more)

### Community 9 - "Run Comparison Bootstrap"
Cohesion: 0.14
Nodes (13): check_runs(), compare(), CompareError, fake_frame(), load_side(), main(), make_fake_run(), _message_has() (+5 more)

### Community 11 - "Micro-batch Probe"
Cohesion: 0.24
Nodes (5): choose_micro_batch(), run_step(), check_attention_kernel(), fake_samples(), find_micro_batch()

### Community 12 - "Cross-Image Transform"
Cohesion: 0.25
Nodes (3): check(), main(), CrossImageTransform

### Community 15 - "Conda Environment"
Cohesion: 0.40
Nodes (4): channels, dependencies, pip, name

## Knowledge Gaps
- **13 isolated node(s):** `Checkpoints last.pt / best.pt`, `colab/run.ipynb thin Colab runner`, `transforms.py coordinate mapping (original <-> resized, flips)`, `data_dir runtime resolution (local/Colab/Kaggle)`, `Tasks T0-T4 with file ownership` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 252 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train()` connect `Run Entry & CLI Config` to `Dataset Loading`, `Coordinate Transforms`, `Pair Batch Checks`, `Flip Augmentation`, `LoRA & EMA Checks`, `Micro-batch Probe`, `On-disk Feature Store`, `Diagnostics Plots`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `train()` (e.g. with `collate_samples()` and `QueryDataset`) actually correct?**
  _`train()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Checkpoints last.pt / best.pt`, `colab/run.ipynb thin Colab runner`, `transforms.py coordinate mapping (original <-> resized, flips)` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Dataset Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.05028305028305028 - nodes in this community are weakly interconnected._
- **Why does `DinoBackbone` connect `DINOv2 Backbone` to `LoRA & EMA Checks`, `Coordinate Transforms`, `Imports & Test Suite`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Should `Coordinate Transforms` be split into smaller, more focused modules?**
  _Cohesion score 0.05945945945945946 - nodes in this community are weakly interconnected._
- **Why does `get_config()` connect `Run Entry & CLI Config` to `Imports & Test Suite`, `Pair Batch Checks`, `Flip Augmentation`, `LoRA & EMA Checks`, `Run Comparison Bootstrap`, `Cross-Image Transform`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._