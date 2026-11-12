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
    "sample_submission_csv": "sample_submission.csv",
    "official_metric_py": "metric.py",  # used for evaluation when present in data_dir
    "val_split_file": "val_pair_ids.json",  # relative to splits_dir

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
    "soft_window": 9,                  # window side in (upsampled) cells
    "soft_temperature": 0.04,
    "feat_upsample": 2,                # bilinear upsampling of the target feature map
    "flip_test": False,                # try a mirrored source and keep the better global alignment
    "feature_cache_size": 256,         # images kept in memory during inference
    "cache_features_to_disk": False,   # frozen mode only: store backbone features in features_dir

    # --- Training (Colab T4; docs/RUN.md section 8) ---
    "train_mode": "finetune",          # last DINOv2 blocks + final LayerNorm ("frozen" needs a head: old runs only)
    "unfreeze_blocks": 2,              # blocks 10 and 11 + final LayerNorm (ViT-B)
    "batch_size": "auto",              # micro-batch: "auto" = largest that fits the GPU x 0.85, or an int
    "effective_batch": 16,             # batch seen by the optimiser (micro-batch x accumulation steps)
    "effective_batch_schedule": None,  # optional ramp {from_epoch: effective_batch}, e.g. {0: 8, 3: 16}
    "epochs": 5,                       # ~127 updates per epoch at effective batch 16 on ~2,040 training queries
    "early_stop_patience": 3,          # stop after this many epochs without a better val PCK; 0 = never
    # Optimiser: AdamW + OneCycleLR; each lr below is the peak (max_lr) of its parameter group.
    "lr_backbone": 1.5e-5,             # top trained block (11) and the final LayerNorm
    "layer_decay": 0.75,               # block i from the top gets lr_backbone * layer_decay**i (block 10: 1.125e-5)
    "warmup_epochs": 1.0,              # OneCycleLR warmup length in epochs (1.0 = the whole first epoch); the peak lr is its last update
    "weight_decay": 0.05,              # AdamW, on every trained parameter
    "grad_clip": 7.0,                  # max total gradient norm
    # Loss (src/models/loss.py explains both):
    #   "soft_ce"  = cross-entropy with Gaussian bumps on ALL accepted targets (v2). Floor > 0, the
    #                probability is spread over every target, so entropy_norm stays around 0.43-0.46.
    #   "marginal" = Lang: -log of the probability on the cells within marginal_radius of ANY target.
    #                The model may commit to one target with a sharp peak: lower entropy. Ignores label_sigma.
    # val/loss of the two is on different scales: compare runs of different losses by val PCK only.
    "loss": "soft_ce",                 # "soft_ce" | "marginal"
    "loss_temperature": 0.04,          # softmax temperature of the loss (both); lower = sharper
    "loss_temperature_start": 0.07,    # anneal linearly from here to loss_temperature over loss_temperature_epochs;
                                       # None = constant loss_temperature
    "loss_temperature_epochs": 1.0,    # length of the temperature ramp in epochs (1.0 = the whole first epoch)
    "label_sigma": 1.0,                # soft_ce: Gaussian label width in patches (0 = one-hot)
    "marginal_radius": 1.0,            # marginal: positive cells within this many cells of a target (Lang: 1.0)
    "checkpoint_every_steps": 0,       # last.pt every N updates; 0 = end of epoch only (fewer Drive writes)
    "val_every_epoch": True,
    # Final submission only: also train on the validation pairs (+22% queries). Validation, best.pt
    # selection and early stop are then off, and best.pt = the weights after the last epoch, so set
    # `epochs` to the best epoch count of a validated run first.
    "train_on_val": False,

    # --- Augmentation (train only; source and target images are augmented separately) ---
    "flip_augment": True,              # Lang ft_pair_flip_aug (both images, keeps left/right consistent)
    "flip_augment_prob": 0.5,
    "aug_rotation_deg": 15.0,          # max rotation; 0 = off
    "aug_rotation_prob": 0.5,
    "aug_color_jitter": 0.3,           # brightness / contrast / saturation strength; 0 = off
    "aug_hue_jitter": 0.05,
    "aug_color_prob": 0.8,
    "aug_grayscale_prob": 0.1,
    "aug_crop_scale": [0.7, 1.0],      # random crop side as a fraction of the image side
    "aug_crop_prob": 0.5,
    "aug_blur_prob": 0.1,
    "aug_edge_margin": 14,             # px in resized space: points closer to the border reject the augmentation
    "aug_max_tries": 3,
    "label_jitter": 0.5,               # std of the noise on target labels, in patches (GeoAware-SC); 0 = off

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


def get_config():
    """A copy of BASE with the paths resolved for the runtime."""
    return resolve_paths(deepcopy(BASE))


def data_path(cfg, key):
    """Absolute path of a file in data_dir, e.g. data_path(cfg, "train_csv")."""
    return Path(cfg["data_dir"]) / cfg[key]


def resolved_device(cfg):
    if cfg["device"] != "auto":
        return cfg["device"]
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
