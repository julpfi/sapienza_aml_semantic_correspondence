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

    # --- Reproducibility ---
    "seed": 0,
    "split_seed": 42,                  # validation split (data/splits.py)
    "val_fraction": 0.2,               # share of image pairs held out for validation
    "max_val_queries": 500,            # cap on validation queries (gives 453)
    "max_train_queries": 0,            # 0 = all (use a small number for quick checks)
    "max_test_queries": 0,             # 0 = all; never limit a real submission
    "deterministic": True,
    "device": "auto",                  # "auto" | "cuda" | "cpu"
    "num_workers": 2,

    # --- Logging ---
    "wandb_project": "sapienza-aml-semantic-correspondence",
    "wandb_entity": "pfingsten-sapienza-universit-di-roma",
    "wandb_mode": "auto",              # "auto" (online if a key is set) | "online" | "offline" | "disabled"
    "publish_best": True,              # upload best.pt to W&B when training ends
    "init_artifact": None,             # start from a teammate's model: "<entity>/<project>/<run_id>-best:latest"

    # --- Model ---
    "model_id": "facebook/dinov2-base",
    "model_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",  # pinned commit on the Hub
    "feature_layer": -1,               # transformer block to read (-1 = last)
    "pos_embed_interp": "hub",         # "hub" = torch.hub interpolation, "hf" = transformers default
    "pos_embed_interp_offset": 0.1,    # torch.hub DINOv2 value (models without registers)
    "image_size": 644,                 # square input; multiple of the patch size (14)
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "attn_implementation": "sdpa",     # transformers attention; "sdpa" uses the memory-efficient kernel on a T4
    # An untrained "linear"/"mlp" head is the identity: without a trained best.pt (--run) the model is
    # the frozen baseline.
    "head": "linear",                  # "none" (identity) | "linear" | "mlp" (needed to train)
    "head_dim": None,                  # output size of the "linear" head; None = backbone width (identity start)

    # --- Matching ---
    "matcher": "soft_window",          # "argmax" | "soft_window"
    "soft_window": 11,                 # window side in (upsampled) cells
    "soft_temperature": 0.02,
    "feat_upsample": 2,                # bilinear upsampling of the target feature map
    "flip_test": False,                # try a mirrored source and keep the better global alignment
    "feature_cache_size": 256,         # images kept in memory during inference
    "cache_features_to_disk": False,   # frozen mode only: store backbone features in features_dir

    # --- Training (Colab T4; why each value: docs/RUN.md section 8) ---
    "train_mode": "finetune",          # "frozen" (train the head) | "finetune" (head + last blocks)
    "unfreeze_blocks": 1,              # last block + final LayerNorm + head
    "batch_size": "auto",              # micro-batch: "auto" = largest that fits the GPU x 0.85, or an int
    "effective_batch": 16,             # batch seen by the optimiser (micro-batch x accumulation steps)
    "effective_batch_schedule": None,  # optional ramp {from_epoch: effective_batch}, e.g. {0: 8, 3: 16}
    "epochs": 5,                       # ~635 updates at effective batch 16
    "early_stop_patience": 2,          # stop after this many epochs without a better val PCK; 0 = never
    # Optimiser: AdamW + OneCycleLR; each lr below is the peak (max_lr) of its parameter group.
    "lr_head": 1e-3,
    "lr_backbone": 1e-5,               # top trained block and the final LayerNorm
    "layer_decay": 0.75,               # block i from the top gets lr_backbone * layer_decay**i
    "lr_pct_start": 0.05,              # OneCycleLR pct_start: share of the updates spent warming up
    "weight_decay": 0.05,              # AdamW, on every trained parameter
    "grad_clip": 5.0,                  # max total gradient norm
    "amp": True,                       # fp16 autocast + GradScaler on CUDA (the T4 has no bf16); ignored on CPU
    "loss_temperature": 0.03,          # softmax temperature of the loss; try 0.02 / 0.05
    "label_sigma": 1.0,                # Gaussian label width in patches (0 = one-hot)
    "checkpoint_every_steps": 150,     # last.pt every N updates (~10 min of ViT-B on a T4); 0 = end of epoch only
    "val_every_epoch": True,
    "peak_mass_window": 5,             # train/peak_mass: probability inside this many cells around the best cell

    # --- Augmentation (train only; source and target images are augmented separately) ---
    # Very light: only aug_image_prob of the images enter the crop/rotation/colour steps below, and a
    # crop removes at most 10% of the side (about 9.5% of the images change).
    "flip_augment": True,              # mirror both images together (keeps left/right consistent)
    "flip_augment_prob": 0.5,
    "aug_image_prob": 0.10,            # share of images that go through the steps below; 0 = off
    "aug_rotation_deg": 10.0,          # max rotation; 0 = off
    "aug_rotation_prob": 0.5,
    "aug_color_jitter": 0.2,           # brightness / contrast / saturation strength; 0 = off
    "aug_hue_jitter": 0.02,
    "aug_color_prob": 0.8,
    "aug_grayscale_prob": 0.05,
    "aug_crop_scale": [0.9, 1.0],      # random crop side as a fraction of the image side
    "aug_crop_prob": 0.5,
    "aug_blur_prob": 0.05,
    "aug_edge_margin": 14,             # px in resized space: points closer to the border reject the augmentation
    "aug_max_tries": 3,
    "label_jitter": 0.25,              # std of the noise on target labels, in patches; 0 = off

    # --- Evaluation ---
    "pck_alpha": 0.1,                  # competition metric: PCK@0.10
    "pck_alphas_report": [0.05, 0.1, 0.15],
    "pck_strict": True,                # correct when d / L < alpha
    "n_visual_checks": 8,              # worst validation queries drawn after evaluate
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

    # --- Submission (CLAUDE.md format) ---
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
