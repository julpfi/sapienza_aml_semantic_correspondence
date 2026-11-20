"""Every setting of the project (BASE) and the runtime paths.

Rules:
- There is one configuration: BASE. To change a setting, edit it here.
- ACTIVE_PRESET (below BASE) is applied on top of BASE by get_config() for new runs; None = BASE as written.
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
    "soft_window": 11,                 # window side in (upsampled) cells (v2: 9; 11 / T 0.02 measured better on run 1)
    "soft_temperature": 0.02,          # v2: 0.04
    "feat_upsample": 2,                # bilinear upsampling of the target feature map
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
    # LoRA (train_mode "lora"): low-rank updates on the attention query/value of the last lora_blocks blocks;
    # everything else frozen. B starts at zero, so update 0 = frozen DINOv2.
    "lora_rank": 8,                    # GECO / common range 8-16
    "lora_alpha": 16,                  # update scale = lora_alpha / lora_rank
    "lora_dropout": 0.05,
    "lora_blocks": 6,                  # last N blocks get LoRA
    "lr_lora": 1e-4,                   # peak lr of the LoRA parameters
    # EMA of the trained weights: validation and best.pt use the averaged weights.
    "ema_decay": 0.99,                 # 0 = off (v2); 0.99 = memory of ~100 updates
    # Stable Diffusion features (frozen, DIFT recipe) fused with the DINOv2 features.
    #   "none"    = DINOv2 only (v2)
    #   "concat"  = normalize(cat(dino, sd_concat_weight * sd)); nothing trained: the frozen gate
    #   "adapter" = normalize(dino + adapter(sd)); the adapter's last layer starts at zero (update 0 = DINOv2)
    "fusion": "concat",                # v2: "none". D-008: concat = DINOv2 fine-tuned on cat(dino, 0.5 * frozen SD)
    "sd_model_id": "sd2-community/stable-diffusion-2-1",  # stabilityai/stable-diffusion-2-1 is no longer public
    "sd_model_revision": "bb2154823665391b4fb29b0b9cf82a198964ee05",  # Hub commit, checked 2026-11-16
    "sd_image_size": 736,              # SD input side; up-block 1 is 1/16 of it: 46 cells = the DINOv2 grid at 644
    "sd_timestep": 261,                # DIFT noising step
    "sd_up_block": 1,                  # decoder block whose output is the feature map (DIFT: 1)
    "sd_concat_weight": 0.5,           # weight of the SD part in "concat"
    "adapter_layers": 2,               # bottleneck blocks after the 1x1 input convolution
    "adapter_dim": 256,                # bottleneck width
    "lr_adapter": 3e-4,                # peak lr of the adapter (the old head at 1e-3 flattened the maps); used with fusion only

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
V3_OFF = {"pair_batch": False, "hard_neg_weight": 1.0, "ema_decay": 0.0, "fusion": "none"}

# v2 values of the training, augmentation and fusion keys that D-008 changed. BASE updated with these = the v2 training
# (run R0 of docs/experiments-sd.md; also what the v2-equivalence checks use). The matching keys stay as in BASE
# (soft_window 11, soft_temperature 0.02), so every run of the matrix is compared at the same matching settings.
V2_TRAINING = {
    "loss": "soft_ce", "epochs": 5, "warmup_epochs": 1.0, "loss_temperature_epochs": 2.0,
    "pair_batch": False, "hard_neg_weight": 1.0, "ema_decay": 0.0, "label_jitter": 0.5,
    "aug_rotation_deg": 15.0, "aug_color_jitter": 0.3, "aug_hue_jitter": 0.05, "aug_color_prob": 0.8,
    "aug_grayscale_prob": 0.1, "aug_crop_scale": [0.7, 1.0], "aug_blur_prob": 0.1, "fusion": "none",
}

# DINOv3 ViT-B/16 (src/models/backbone_dinov3.py). BASE updated with these = the same pipeline on DINOv3.
# Patch 16: 736 px gives the 46 x 46 grid of DINOv2 at 644, so every setting counted in cells keeps its meaning;
# aug_edge_margin is one patch (16 / 736 = 14 / 644 of the image side). The weights are gated on the Hub:
# accept the DINOv3 license with the account of HF_TOKEN. Compare with a DINOv2 run: compare_runs.py --allow image_size.
DINOV3_B16 = {
    "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "model_revision": "5931719e67bbdb9737e363e781fb0c67687896bc",  # Hub commit, checked 2026-11-19
    "image_size": 736,
    "aug_edge_margin": 16,
}

# First DINOv3 run (decisions.md D-010): DINOV3_B16 + the D-008 training of BASE, without SD fusion.
# - Training keys stay as in BASE (finetune blocks 10-11 + norm, lr_backbone 1.5e-5, layer_decay 0.75, marginal loss
#   0.07 -> 0.04, pair batch 4, hard negatives 3.0, EMA 0.99, 8 epochs). Same 46 x 46 grid and the same token count
#   (2,116 patches + 5 prefix tokens vs + 1), so the cell-based settings, the memory and the auto micro-batch carry over.
#   A DINOv2 twin for compare_runs.py is BASE with only fusion "none" (+ --allow image_size).
# - fusion "none": measures the backbone alone; SD would cost a second fp16 network and mix two effects.
#   Add "concat" back only after DINOv3 beats DINOv2.
# - Sweep (evaluate only, report): DINOv3 cosine maps may peak differently from DINOv2, so the window / temperature
#   are swept again, and DINOv3 is said to improve with resolution (RoPE, no position table), so larger inputs are
#   scored too (multiples of 16; a larger input = a finer grid, so the cell-based windows cover less of the image).
DINOV3_FIRST_RUN = {
    **DINOV3_B16,
    "fusion": "none",
    "sweep_soft_window": [7, 9, 11, 13],
    "sweep_soft_temperature": [0.01, 0.02, 0.04],
    "sweep_image_size": [736, 896, 1024],
}

# DINOv3 fine-tuning at the best frozen sweep setting (run 20261120-104658_evaluate, decisions.md D-011):
# 896 px + flip_test + window 7 / T 0.02 = 35.5 val PCK@0.10 frozen (24.7 at 736 / no flip / window 11).
# - 896 px = 56 x 56 cells (DINOv2 at 644: 46 x 46). Cell-based training keys (marginal_radius, label_jitter,
#   hard_neg_*) are kept in cells, so they are ~18% tighter in image share; aug_edge_margin stays one patch (16 px).
#   ~1.5x the tokens of 736: slower epochs and a smaller auto micro-batch (still 4 pairs per update).
# - flip_test is also used by the per-epoch validation, so best.pt is chosen with the matching of the submission.
#   It mirrors the source when that aligns better: it works around DINOv3's positional bias (left/right) at inference.
# - Training keys stay as in BASE (D-008), fusion "none". The frozen-sweep window may be too wide or narrow after
#   fine-tuning (sharper peaks), so the evaluate sweep re-scores windows / temperatures around it.
# - Not directly comparable with runs at other matching settings: compare_runs.py needs --allow for the differing
#   keys, or re-evaluate the reference at the same settings.
DINOV3_FT_896 = {
    **DINOV3_B16,
    "image_size": 896,
    "fusion": "none",
    "flip_test": True,
    "soft_window": 7,
    "soft_temperature": 0.02,
    "sweep_flip_test": [True],
    "sweep_soft_window": [5, 7, 9, 11],
    "sweep_soft_temperature": [0.01, 0.02, 0.04],
    "sweep_image_size": [],            # trained at 896: score that size only
}

# Settings that get_config() applies on top of BASE, for new runs only (train, and evaluate / submit without --run).
# None = BASE as written. Saved runs never see it: --run / --resume load their own config.json (missing keys from
# BASE), so DINOv2 runs keep loading exactly as before.
ACTIVE_PRESET = DINOV3_FT_896


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
    """A copy of BASE, updated with ACTIVE_PRESET, with the paths resolved for the runtime."""
    cfg = deepcopy(BASE)
    if ACTIVE_PRESET:
        cfg.update(deepcopy(ACTIVE_PRESET))
    return resolve_paths(cfg)


def data_path(cfg, key):
    """Absolute path of a file in data_dir, e.g. data_path(cfg, "train_csv")."""
    return Path(cfg["data_dir"]) / cfg[key]


def resolved_device(cfg):
    if cfg["device"] != "auto":
        return cfg["device"]
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
