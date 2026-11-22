"""Every setting of the project (BASE) and the runtime paths.

Rules:
- There is one configuration: BASE. To change a setting, edit it here.
- Every run saves the full config to <output_dir>/<run_id>/config.json.
- Paths are None in BASE and are filled in for the detected runtime (local, Colab or Kaggle).
  The SC_* environment variables override them.
"""

import os
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Runtime detection and paths
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive/aml_semantic_correspondence")

RUNTIME_PATHS = {
    "local": {
        "data_dir": REPO_ROOT / "data" / "raw",
        "output_dir": REPO_ROOT / "outputs",
        "weights_dir": REPO_ROOT / "weights",
        "features_dir": REPO_ROOT / "data" / "features",
        "splits_dir": REPO_ROOT / "splits",
    },
    "colab": {
        # Data is re-downloaded to the fast local disk each session; runs and weights go to Drive.
        "data_dir": Path("/content/data/raw"),
        "output_dir": DRIVE_ROOT / "outputs",
        "weights_dir": DRIVE_ROOT / "weights",
        "features_dir": DRIVE_ROOT / "features",
        "splits_dir": REPO_ROOT / "splits",
    },
    "kaggle": {
        # Attach the competition data and a dataset holding weights/<model folder>.
        "data_dir": Path("/kaggle/input/competitions/aml-challenge-2026-27"),
        "output_dir": Path("/kaggle/working/outputs"),
        "weights_dir": Path("/kaggle/input/dinov2-weights"),
        "features_dir": Path("/kaggle/working/features"),
        # Kaggle inputs are read-only; the notebook copies the tracked split here.
        "splits_dir": Path("/kaggle/working/splits"),
    },
}

# Environment variables that override the paths above.
PATH_ENV_VARS = {
    "data_dir": "SC_DATA_DIR",
    "output_dir": "SC_OUTPUT_DIR",
    "weights_dir": "SC_WEIGHTS_DIR",
    "features_dir": "SC_FEATURES_DIR",
    "splits_dir": "SC_SPLITS_DIR",
}


def detect_runtime():
    """Return "kaggle", "colab" or "local"."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return "colab"
    return "local"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE = {
    # --- Runtime and paths (None = resolved for the runtime, see RUNTIME_PATHS) ---
    "runtime": "auto",                 # "auto" | "local" | "colab" | "kaggle"
    "competition": "aml-challenge-2026-27",
    "data_dir": None,                  # folder with train.csv, test.csv, images/
    "output_dir": None,                # run folders: <output_dir>/<run_id>/
    "weights_dir": None,               # local Hugging Face snapshots: <weights_dir>/<model folder>
    "features_dir": None,              # on-disk cache of frozen backbone features
    "splits_dir": None,                # tracked validation split (splits/ in the repo)
    "train_csv": "train.csv",          # relative to data_dir
    "test_csv": "test.csv",
    "train_pairs_json": "train_pairs.json",
    "test_pairs_json": "test_pairs.json",
    "sample_submission_csv": "sample_submission.csv",
    "official_metric_py": "metric.py",  # used for evaluation when present in data_dir
    "val_split_file": "val_pair_ids.json",  # relative to splits_dir
    "val_folds_file": "val_disjoint_folds.json",  # image-disjoint connected-component folds (splits_dir)
    "val_fold": None,                  # None = legacy val_pair_ids.json; 0..val_n_folds-1 = image-disjoint fold
    "val_n_folds": 5,                  # number of image-disjoint folds in val_folds_file

    # Competition CSV columns -> names used in the code.
    "csv_columns": {
        "query_id": "id",
        "pair_id": "pair_id",
        "src_image": "source_image",
        "trg_image": "target_image",
        "src_x": "source_x",
        "src_y": "source_y",
        "trg_xys": "target_points",    # JSON list of accepted [x, y] targets
        "scale": "scale",              # object size L used by PCK
        "category": "category",        # not in the challenge data (checked 2026-11-04): everything is "all"
    },

    # --- Reproducibility (Lang values) ---
    "seed": 0,                         # Lang: 0 (main had 1019)
    "split_seed": 42,                  # Lang notebook
    "val_fraction": 0.2,               # share of image pairs held out for validation
    "max_val_queries": 500,            # Lang notebook cap (gave 453 queries)
    "max_train_queries": 0,            # 0 = all (use a small number for quick checks)
    "max_test_queries": 0,             # 0 = all; never limit a real submission
    "deterministic": True,
    "device": "auto",                  # "auto" | "cuda" | "cpu"
    "num_workers": 2,

    # --- Logging (main) ---
    "wandb_project": "sapienza-aml-semantic-correspondence",
    "wandb_entity": "pfingsten-sapienza-universit-di-roma",
    "wandb_mode": "auto",              # "auto" (online if a key is set) | "online" | "offline" | "disabled"
    "publish_best": True,              # upload best.pt to W&B when training ends
    "init_artifact": None,             # start from a teammate's model: "<entity>/<project>/<run_id>-best:latest"

    # --- Model (Lang: torch.hub dinov2_vitb14 -> facebook/dinov2-base) ---
    "model_id": "facebook/dinov2-base",
    "model_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",  # pinned commit on the Hub
    "feature_layer": -1,               # transformer block to read (-1 = last); Lang `layer`
    "feature_layers": None,            # None = single feature_layer; e.g. [-3, -1] = L2-norm concat of multiple blocks
    "pos_embed_interp": "hub",         # "hub" = torch.hub interpolation (Lang results), "hf" = transformers default
    "pos_embed_interp_offset": 0.1,    # torch.hub DINOv2 value (models without registers)
    "image_size": 644,                 # square input; multiple of the patch size (14). Lang submission: 644
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "attn_implementation": "sdpa",     # transformers attention; "sdpa" uses the memory-efficient kernel on a T4
    # No trained head (as in Lang): the matching features are the DINOv2 features themselves, so
    # without a trained best.pt (--run) the model is the frozen baseline. Runs trained with a head keep
    # loading through their own config.json (head, head_dim, lr_head).
    "head": "none",

    # --- Matching (Lang) ---
    "matcher": "soft_window",          # "argmax" | "soft_window"
    "soft_window": 11,                 # window side in (upsampled) cells (v2: 9; 11 / T 0.02 measured better on run 1)
    "soft_temperature": 0.02,          # v2: 0.04
    "feat_upsample": 2,                # bilinear upsampling of the target feature map
    "csls_weight": 0.0,                # >0 subtracts CSLS hubness baseline (top-k mean src sim per trg cell)
    "csls_k": 10,                      # number of top source patches averaged for CSLS hubness
    "flip_test": False,                # try a mirrored source and keep the better global alignment
    "feature_cache_size": 256,         # images kept in memory during inference
    "cache_features_to_disk": False,   # frozen mode only: store backbone features in features_dir

    # --- Training (Colab T4; docs/RUN.md section 8) ---
    "train_mode": "finetune",          # last DINOv2 blocks + final LayerNorm ("frozen" needs a head: old runs only;
                                       # "lora" = LoRA on the last lora_blocks blocks, see the v3 additions)
    "unfreeze_blocks": 2,              # blocks 10 and 11 + final LayerNorm (ViT-B)
    "batch_size": "auto",              # micro-batch: "auto" = largest that fits the GPU x 0.85, or an int
    "effective_batch": 16,             # batch seen by the optimiser (micro-batch x accumulation steps)
    "effective_batch_schedule": None,  # optional ramp {from_epoch: effective_batch}, e.g. {0: 8, 3: 16}
    "epochs": 8,                       # v2: 5. Pair mode: ~78 updates per epoch at pair_effective_batch 4 (~127 per-query)
    "early_stop_patience": 3,          # stop after this many epochs without a better val PCK; 0 = never
    # Optimiser: AdamW + OneCycleLR; each lr below is the peak (max_lr) of its parameter group.
    "lr_backbone": 1.5e-5,             # top trained block (11) and the final LayerNorm
    "layer_decay": 0.75,               # block i from the top gets lr_backbone * layer_decay**i (block 10: 1.125e-5)
    "warmup_epochs": 1.5,              # v2: 1.0. OneCycleLR warmup length in epochs; the peak lr is its last update
    "weight_decay": 0.05,              # AdamW, on every trained parameter
    "grad_clip": 10.0,                 # max total gradient norm
    # Loss (src/models/loss.py explains both):
    #   "soft_ce"  = cross-entropy with Gaussian bumps on ALL accepted targets (v2). Floor > 0, the
    #                probability is spread over every target, so entropy_norm stays around 0.43-0.46.
    #   "marginal" = Lang: -log of the probability on the cells within marginal_radius of ANY target.
    #                The model may commit to one target with a sharp peak: lower entropy. Ignores label_sigma.
    # val/loss of the two is on different scales: compare runs of different losses by val PCK only.
    "loss": "marginal",                # "soft_ce" (v2) | "marginal"
    "loss_temperature": 0.04,          # softmax temperature of the loss (both); lower = sharper
    "loss_temperature_start": 0.07,    # anneal linearly from here to loss_temperature over loss_temperature_epochs;
                                       # None = constant loss_temperature
    "loss_temperature_epochs": 3.0,    # v2: 2.0. Length of the temperature ramp in epochs (3.0 = bottom reached at the end of epoch 2)
    "label_sigma": 1.0,                # soft_ce: Gaussian label width in patches (0 = one-hot)
    "marginal_radius": 1.0,            # marginal: positive cells within this many cells of a target (Lang: 1.0)
    "loss_upsample": 1,                # >1 upsamples trg_feats before loss (2 = 92x92, eliminates 0-cell queries)
    "adaptive_pos_radius": False,      # True = cap marginal positive radius at 0.08 * object_size_in_cells
    "bidirectional_loss_weight": 0.0,  # >0 adds reverse trg->src marginal loss on single-target queries (pair_batch)
    "checkpoint_every_steps": 0,       # last.pt every N updates; 0 = end of epoch only (fewer Drive writes)
    "val_every_epoch": True,
    # Final submission only: also train on the validation pairs (+22% queries). Validation, best.pt
    # selection and early stop are then off, and best.pt = the weights after the last epoch, so set
        # `epochs` to the best epoch count of a validated run first.
    "train_on_val": False,

    # --- v3 additions (docs/PLAN.md). BASE holds the recommended stage-1 setting (decisions.md D-008): pair batch,
    # hard negatives, EMA, marginal loss, plus fusion "concat". v2 = BASE updated with V2_TRAINING (below BASE). ---
    # Pair-batched training: one item = one image pair with all its queries, one backbone pass per pair.
    "pair_batch": True,                # False = one item per query (v2)
    "pair_effective_batch": 4,         # pairs per optimiser update (~6 queries per pair, so ~25 queries)
    # Hard negatives (needs pair_batch): cells near the accepted targets of the OTHER queries of the pair
    # count hard_neg_weight times in the softmax (+log(weight) on their logit). Aims at wrong_keypoint errors.
    "hard_neg_weight": 3.0,            # 1.0 = off (v2); try 2.0 / 4.0 around it
    "hard_neg_radius": 1.0,            # cells around another query's target that are negatives
    "hard_neg_exclusion": 2.0,         # cells around the query's own accepted targets that are never negatives
    # LoRA (train_mode "lora"): low-rank updates on selected attention/MLP linears of the last lora_blocks blocks;
    # everything else frozen. B starts at zero, so update 0 = frozen DINOv2.
    "lora_rank": 8,                    # GECO / common range 8-16
    "lora_alpha": 16,                  # update scale = lora_alpha / lora_rank
    "lora_dropout": 0.05,
    "lora_blocks": 6,                  # last N blocks get LoRA
    "lora_targets": ["q_proj", "v_proj"],  # linear layers wrapped by LoRA; lorav2 uses attention + MLP
    "lora_train_layernorm": False,     # True = also train LayerNorms in last lora_blocks + final norm (lorav2)
    "lr_lora": 1e-4,                   # peak lr of the LoRA parameters
    # EMA of the trained weights: validation and best.pt use the averaged weights.
    "ema_decay": 0.99,                 # 0 = off (v2); 0.99 = memory of ~100 updates
    # Stable Diffusion features (frozen, DIFT recipe) fused with the DINOv2 features.
    #   "none"    = DINOv2 only (v2)
    #   "concat"  = normalize(cat(dino, sd_concat_weight * sd)); nothing trained: the frozen gate
    #   "adapter" = normalize(dino + adapter(sd)); the adapter's last layer starts at zero (update 0 = DINOv2)
    "fusion": "none",                  # "none" (default on giuseppe branch) | "concat" | "adapter"
    "sd_model_id": "sd2-community/stable-diffusion-2-1",  # stabilityai/stable-diffusion-2-1 is no longer public
    "sd_model_revision": "bb2154823665391b4fb29b0b9cf82a198964ee05",  # Hub commit, checked 2026-11-16
    "sd_image_size": 736,              # SD input side; up-block 1 is 1/16 of it: 46 cells = the DINOv2 grid at 644
    "sd_timestep": 261,                # DIFT noising step
    "sd_up_block": 1,                  # decoder block whose output is the feature map (DIFT: 1)
    "sd_concat_weight": 0.5,           # weight of the SD part in "concat"
    "adapter_layers": 2,               # bottleneck blocks after the 1x1 input convolution
    "adapter_dim": 256,                # bottleneck width
    "lr_adapter": 3e-4,                # peak lr of the adapter (the old head at 1e-3 flattened the maps); used with fusion only

    # --- Transformation-aware additions (branch giuseppe) ---
    # 1. Cross-image spatial context & multi-scale geometric adapter (src/models/cross_transform.py):
    #    trainable multi-scale dilated conv + bidirectional cross-attention between source and target feature maps.
    "cross_transform": True,           # True = train & apply CrossImageTransform; False = off
    "cross_dim": 256,                  # internal bottleneck width of the multi-scale + cross-attention adapter
    "cross_heads": 4,                  # attention heads (cross_dim must be divisible by cross_heads)
    "cross_dropout": 0.05,
    "lr_cross": 3e-4,                  # peak lr of the CrossImageTransform parameter group
    # 2. Two-stage object-centric zoom at inference (src/models/zoom.py):
    #    Stage 1 matches all source_xy context points on the full image; Stage 2 crops around the
    #    object on both images (+ zoom_margin) and re-matches at full image_size.
    "object_zoom": False,              # True = enable 2-stage object-centric zoom at inference
    "zoom_margin": 0.25,               # relative margin around the 5th-95th percentile point box
    "zoom_min_frac": 0.20,             # minimum crop side as a fraction of the image side
    "zoom_max_area": 1.0,              # skip cropping a side whose box already covers > this fraction
    "zoom_stage1_weight": 0.0,         # blend weight of Stage-1 full-image similarity into Stage-2 crop
    "zoom_pct_low": 5.0,               # lower percentile of coarse points (ignores outliers)
    "zoom_pct_high": 95.0,             # upper percentile of coarse points

    # --- Augmentation (train only; source and target images are augmented separately) ---
    "flip_augment": True,              # Lang ft_pair_flip_aug (both images, keeps left/right consistent)
    "flip_augment_prob": 0.5,
    "aug_rotation_deg": 10.0,          # max rotation; 0 = off (v2: 15)
    "aug_rotation_prob": 0.5,
    "aug_color_jitter": 0.2,           # brightness / contrast / saturation strength; 0 = off (v2: 0.3)
    "aug_hue_jitter": 0.02,            # v2: 0.05
    "aug_color_prob": 0.5,             # v2: 0.8
    "aug_grayscale_prob": 0.05,        # v2: 0.1
    "aug_crop_scale": [0.8, 1.0],      # random crop side as a fraction of the image side (v2: [0.7, 1.0])
    "aug_crop_prob": 0.5,
    "aug_blur_prob": 0.05,             # v2: 0.1
    "aug_edge_margin": 14,             # px in resized space: points closer to the border reject the augmentation
    "aug_max_tries": 3,
    "label_jitter": 0.25,              # std of the noise on target labels, in patches (GeoAware-SC); 0 = off (v2: 0.5)

    # --- Evaluation ---
    "pck_alpha": 0.1,                  # competition metric: PCK@0.10
    "pck_alphas_report": [0.05, 0.1, 0.15],
    "pck_strict": True,                # correct when d / L < alpha (Lang notebook)
    "n_visual_checks": 4,              # worst validation queries drawn on W&B (val/worst_predictions)
    "visual_thumb_px": 320,            # longest side of the images drawn in that plot
    "near_miss_factor": 2.0,           # near miss: alpha <= d/L < factor * alpha
    "far_miss_dist": 0.5,              # far miss: d/L above this
    "edge_bins": [0.05, 0.15],         # distance of the nearest target to the image border / L
    "wandb_evaluate": True,            # also log evaluate runs to W&B (job type "evaluate")

    # --- Sweeps (evaluate only; a report, never used by the submission). Empty list = skip. ---
    # Features are computed once per (model_id, image_size) and reused for all matching settings.
    "sweep_flip_test": [False, True],
    "sweep_soft_window": [3, 5, 7, 9, 11],
    "sweep_soft_temperature": [0.02, 0.04],
    "sweep_image_size": [],            # e.g. [644, 756, 840, 952] (multiples of 14); [] = image_size only
    "sweep_model_id": [],              # e.g. ["facebook/dinov2-base", "facebook/dinov2-large"]; [] = model_id only
    "sweep_model_revisions": {         # pinned Hub commits of the other sweep models (model_id uses model_revision)
        "facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",  # Hub commit, checked 2026-11-04
    },

    # --- Submission (Lang / CLAUDE.md format) ---
    "submission_columns": ["id", "prediction"],
    "prediction_decimals": 2,          # "[75.81,228.38]"
    "clip_to_image": True,
    "kaggle_submission_path": "/kaggle/working/submission.csv",  # copied there by --mode submit on Kaggle
}


# "Off" values of the v3 switches. A run saved before these keys existed gets them, not the current BASE
# values: BASE may have a switch on for a new experiment while an old run is evaluated or resumed.
V3_OFF = {
    "pair_batch": False, "hard_neg_weight": 1.0, "ema_decay": 0.0, "fusion": "none",
    "cross_transform": False, "object_zoom": False,
    "loss_upsample": 1, "adaptive_pos_radius": False, "lora_train_layernorm": False,
    "val_fold": None, "feature_layers": None, "csls_weight": 0.0, "bidirectional_loss_weight": 0.0,
}

# v2 values of the training, augmentation and fusion keys that D-008 changed. BASE updated with these = the v2 training
# (run R0 of docs/experiments-sd.md; also what the v2-equivalence checks use). The matching keys stay as in BASE
# (soft_window 11, soft_temperature 0.02), so every run of the matrix is compared at the same matching settings.
V2_TRAINING = {
    "loss": "soft_ce", "epochs": 5, "warmup_epochs": 1.0, "loss_temperature_epochs": 2.0,
    "pair_batch": False, "hard_neg_weight": 1.0, "ema_decay": 0.0, "label_jitter": 0.5,
    "aug_rotation_deg": 15.0, "aug_color_jitter": 0.3, "aug_hue_jitter": 0.05, "aug_color_prob": 0.8,
    "aug_grayscale_prob": 0.1, "aug_crop_scale": [0.7, 1.0], "aug_blur_prob": 0.1, "fusion": "none",
    "cross_transform": False, "object_zoom": False,
    "loss_upsample": 1, "adaptive_pos_radius": False, "lora_train_layernorm": False,
    "val_fold": None, "feature_layers": None, "csls_weight": 0.0, "bidirectional_loss_weight": 0.0,
}

# Unified winners preset: lorav2 attention+MLP LoRA (less overfitting on unseen test images) +
# analysis_report 92x92 upsampled loss & scale-adaptive positive radius +
# wider scale/rotation augmentation + 2-stage object-centric zoom at inference.
UNIFIED_LORA = {
    "train_mode": "lora",
    "lora_blocks": 4,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
    "lora_train_layernorm": True,
    "lr_lora": 1e-4,
    "epochs": 6,
    "warmup_epochs": 1.0,
    "loss": "marginal",
    "loss_upsample": 2,
    "adaptive_pos_radius": True,
    "loss_temperature": 0.03,
    "loss_temperature_start": 0.06,
    "loss_temperature_epochs": 2.0,
    "hard_neg_weight": 5.0,
    "aug_rotation_deg": 25.0,
    "aug_crop_scale": [0.45, 1.0],
    "cross_transform": False,
    "object_zoom": True,
    "fusion": "none",
}

# v2 of UNIFIED_LORA: preserves the 0.64851 recipe and adds:
# 1. Image-disjoint 5-fold validation (0% image leakage between train and val)
# 2. Multi-layer DINOv2 readout (blocks -3 and -1 concatenated, 1536-d)
# 3. Bidirectional trg->src marginal loss on single-target queries
# (CSLS disabled: top-k hubness on dense zoomed maps penalized foreground object peaks by -4.6 pts)
UNIFIED_LORA_V2 = {
    **UNIFIED_LORA,
    "val_fold": 0,
    "max_val_queries": 0,
    "feature_layers": [-3, -1],
    "csls_weight": 0.0,
    "csls_k": 10,
    "bidirectional_loss_weight": 0.5,
}

PRESETS = {
    "unified_lora": UNIFIED_LORA,
    "unified_lora_v2": UNIFIED_LORA_V2,
}
ACTIVE_PRESET = None


def resolve_paths(cfg):
    """Fill in path settings left as None for the detected runtime. Returns cfg."""
    runtime = detect_runtime() if cfg["runtime"] == "auto" else cfg["runtime"]
    if runtime not in RUNTIME_PATHS:
        raise ValueError(f"Unknown runtime {runtime!r}. Use one of {sorted(RUNTIME_PATHS)}.")
    defaults = dict(RUNTIME_PATHS[runtime])
    if runtime == "colab" and not DRIVE_ROOT.parent.is_dir():
        # Drive not mounted: keep runs on the local disk (lost when the session ends).
        defaults.update(output_dir=Path("/content/outputs"), weights_dir=Path("/content/weights"),
                        features_dir=Path("/content/features"))
    cfg["runtime"] = runtime
    for key, default in defaults.items():
        if cfg.get(key) is None:
            cfg[key] = os.environ.get(PATH_ENV_VARS[key]) or str(default)
    return cfg


def get_config(preset=None):
    """A copy of BASE (optionally updated with a named preset) with paths resolved for the runtime."""
    cfg = deepcopy(BASE)
    chosen = preset or os.environ.get("SC_PRESET") or ACTIVE_PRESET
    if chosen and chosen != "none":
        norm = chosen.replace("-", "_")
        if norm not in PRESETS:
            raise ValueError(f"Unknown preset {chosen!r}. Use one of {sorted(PRESETS)} or 'none'.")
        cfg.update(deepcopy(PRESETS[norm]))
    return resolve_paths(cfg)


def data_path(cfg, key):
    """Absolute path of a file in data_dir, e.g. data_path(cfg, "train_csv")."""
    return Path(cfg["data_dir"]) / cfg[key]


def resolved_device(cfg):
    if cfg["device"] != "auto":
        return cfg["device"]
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
