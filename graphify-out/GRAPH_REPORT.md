# Graph Report - advance  (2026-11-24)

## Corpus Check
- 55 files · ~39,259 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: (none) 3, .ipynb 2, .example 1)

## Summary
- 647 nodes · 1509 edges · 32 communities (21 shown, 11 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.9)
- Token cost: 81,173 input · 0 output

## Graph Freshness
- Built from commit: `4981c2f5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evaluate.py
- torch
- run.py
- Imports & Test Suite
- Docs: Data & Decisions
- Pair Batch Checks
- pathlib
- Flip Augmentation
- check_lora_ema.py
- Run Comparison Bootstrap
- Ensemble Model
- Micro-batch Probe
- Cross-Image Transform
- On-disk Feature Store
- Diagnostics Plots
- Conda Environment
- train.py
- sweep.py
- json
- preprocess_image
- PCKMeter
- models/__init__.py
- CorrespondenceModel
- datasets.py
- Findings
- Review: D-009 SD multi-block fusion for DINOv2-large (2026-11-24)
- predict_with_zoom
- transforms.py

## God Nodes (most connected - your core abstractions)
1. `train()` - 43 edges
2. `RUN.md run guide` - 28 edges
3. `main()` - 23 edges
4. `get_config()` - 22 edges
5. `main()` - 19 edges
6. `build_dataset()` - 17 edges
7. `predict_with_zoom()` - 16 edges
8. `Plan: orientation-aware training + LoRA/EMA + SD fusion on branch sd` - 16 edges
9. `read_queries()` - 14 edges
10. `evaluate()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Verified without change` --references--> `Fusion`  [INFERRED]
  docs/reviews/2026-11-24-sd-dinov2-large-fusion.md → src/models/fusion.py
- `Not changed (needs CEO decision)` --references--> `_StopForward`  [INFERRED]
  docs/reviews/2026-11-24-sd-dinov2-large-fusion.md → src/models/sd.py
- `Verified without change` --references--> `_StopForward`  [INFERRED]
  docs/reviews/2026-11-24-sd-dinov2-large-fusion.md → src/models/sd.py
- `train()` --uses--> `QueryDataset`  [INFERRED]
  src/train.py → data/datasets.py
- `check_v2_equivalence()` --indirect_call--> `collate_samples()`  [INFERRED]
  scripts/check_pair_batch.py → data/datasets.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Inference matching pipeline** — docs_run_dinov2_backbone, docs_run_concat_fusion, docs_run_soft_argmax, docs_run_pipeline [EXTRACTED 1.00]
- **v3 additions targeting near-miss / wrong-keypoint errors** — docs_plan_pair_batch, docs_plan_hard_negatives, docs_plan_lora, docs_plan_ema, docs_plan_sd_features, docs_plan_fusion_adapter [EXTRACTED 1.00]

## Communities (32 total, 11 thin omitted)

### Community 0 - "evaluate.py"
Cohesion: 0.16
Nodes (8): build_dataset(), read_queries(), evaluate(), validation_loss(), edge_distance(), normalized_distance(), official_score(), target_group()

### Community 1 - "torch"
Cohesion: 0.19
Nodes (8): to_original(), cell_centers(), compute_similarity_map(), csls_hubness(), locate(), predict_from_features(), bbox_crop(), _sample_stage1_into_crop()

### Community 2 - "run.py"
Cohesion: 0.08
Nodes (20): _apply_cli_overrides(), load_config(), load_trained_weights(), main(), parse_args(), preflight(), data_path(), log_evaluation() (+12 more)

### Community 3 - "Imports & Test Suite"
Cohesion: 0.08
Nodes (19): install(), load_colab_secrets(), main(), mount_drive(), pip_requirements(), pull(), run(), version_check() (+11 more)

### Community 4 - "Docs: Data & Decisions"
Cohesion: 0.06
Nodes (49): Data README, csv_columns mapping, data_dir runtime resolution (local/Colab/Kaggle), Lang notebook, QueryDataset (one CSV row = one query), scale = object size L (longest bbox side), target_points (all accepted targets), transforms.py coordinate mapping (original <-> resized, flips) (+41 more)

### Community 5 - "Pair Batch Checks"
Cohesion: 0.07
Nodes (26): base_cfg(), cell_xy(), check_augment_pair(), check_hard_negatives(), check_matching_loss_unchanged(), check_pair_batch_loss(), check_pair_dataset(), check_pair_loss_equals_matching_loss() (+18 more)

### Community 6 - "pathlib"
Cohesion: 0.06
Nodes (10): main(), resolved_device(), DinoBackbone, _hub_interpolate_pos_encoding(), local_model_dir(), ProjectionHead, _load(), local_sd_dir() (+2 more)

### Community 7 - "Flip Augmentation"
Cohesion: 0.13
Nodes (15): flip_points(), flip_sample(), main(), run(), _augment_image(), augment_pair(), augment_sample(), _flip_pair() (+7 more)

### Community 8 - "check_lora_ema.py"
Cohesion: 0.06
Nodes (22): check(), main(), make_cfg(), raises(), randomise_B(), wrap_once(), check(), head_model_class() (+14 more)

### Community 9 - "Run Comparison Bootstrap"
Cohesion: 0.12
Nodes (13): check_runs(), compare(), CompareError, fake_frame(), load_side(), main(), make_fake_run(), _message_has() (+5 more)

### Community 11 - "Micro-batch Probe"
Cohesion: 0.24
Nodes (5): choose_micro_batch(), run_step(), check_attention_kernel(), fake_samples(), find_micro_batch()

### Community 12 - "Cross-Image Transform"
Cohesion: 0.19
Nodes (4): check(), main(), CrossImageTransform, _sinusoidal_2d()

### Community 15 - "Conda Environment"
Cohesion: 0.40
Nodes (4): channels, dependencies, pip, name

### Community 16 - "train.py"
Cohesion: 0.09
Nodes (14): collate_samples(), augmentation_enabled(), averaged_weights(), configure_trainable(), effective_batch_for(), loss_temperature_at(), plan_updates(), _rng_state() (+6 more)

### Community 20 - "sweep.py"
Cohesion: 0.18
Nodes (6): alignment_score(), choose_source(), feature_config(), run_sweep(), _score_setting(), sweep_grid()

### Community 21 - "json"
Cohesion: 0.23
Nodes (6): disjoint_folds_path(), load_disjoint_folds(), load_val_pair_ids(), make_disjoint_folds(), make_val_pair_ids(), split_path()

### Community 22 - "preprocess_image"
Cohesion: 0.18
Nodes (7): preprocess_image(), to_resized(), _backbone_features(), batch_loss(), compute_features(), jitter_labels(), pair_batch_loss()

### Community 23 - "PCKMeter"
Cohesion: 0.23
Nodes (3): confidence_interval(), is_correct(), PCKMeter

### Community 27 - "Findings"
Cohesion: 0.17
Nodes (11): Findings, Literature review: Stable Diffusion + DINOv2 (incl. ViT-L) fusion for semantic correspondence (2026-11-24), Mapping to this repo, Open questions / unsure, Proposed experiments, Q1. SD+DINO architecture [1] and GeoAware-SC code [2b], Q2. Numbers (SPair-71k, PCK@0.10 bbox), Q3. GeoAware-SC supervised recipe [2], [2b] (+3 more)

### Community 28 - "Review: D-009 SD multi-block fusion for DINOv2-large (2026-11-24)"
Cohesion: 0.20
Nodes (8): Checks, How to undo, Modifications, Not changed (needs CEO decision), Review: D-009 SD multi-block fusion for DINOv2-large (2026-11-24), Skipped (do-not-touch list or unsure), Verified without change, _StopForward

### Community 29 - "predict_with_zoom"
Cohesion: 0.25
Nodes (4): load_pair_source_xy(), predict_with_zoom(), _similarity_map(), _stage1_pair()

## Knowledge Gaps
- **13 isolated node(s):** `data_dir runtime resolution (local/Colab/Kaggle)`, `transforms.py coordinate mapping (original <-> resized, flips)`, `Resource request (diffusers, Hub lookups, tiny SD, subagents)`, `Tasks T0-T4 with file ownership`, `entropy_norm chart` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 269 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train()` connect `train.py` to `evaluate.py`, `torch`, `run.py`, `Imports & Test Suite`, `Pair Batch Checks`, `Flip Augmentation`, `check_lora_ema.py`, `Micro-batch Probe`, `On-disk Feature Store`, `Diagnostics Plots`, `json`, `preprocess_image`, `datasets.py`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `train()` (e.g. with `collate_samples()` and `QueryDataset`) actually correct?**
  _`train()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `data_dir runtime resolution (local/Colab/Kaggle)`, `transforms.py coordinate mapping (original <-> resized, flips)`, `Resource request (diffusers, Hub lookups, tiny SD, subagents)` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08013937282229965 - nodes in this community are weakly interconnected._
- **Why does `DinoBackbone` connect `pathlib` to `check_lora_ema.py`, `models/__init__.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Should `Imports & Test Suite` be split into smaller, more focused modules?**
  _Cohesion score 0.08080808080808081 - nodes in this community are weakly interconnected._
- **Why does `SDFeatures` connect `pathlib` to `models/__init__.py`, `CorrespondenceModel`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._