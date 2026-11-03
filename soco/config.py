"""Central configuration. Every experiment is one Config, so runs are comparable."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---------------- data ----------------
    data_root: str = "/kaggle/input/spair-71k/SPair-71k"
    data_format: str = "spair"          # "spair" | "csv"
    split: str = "val"                  # "trn" | "val" | "test"
    csv_path: str = ""                  # used when data_format == "csv"
    image_dir: str = ""                 # used when data_format == "csv"
    max_pairs: int = 0                  # 0 = all pairs (use a small number for quick checks)
    categories: list[str] = field(default_factory=list)  # empty = all

    # ---------------- backbone ----------------
    # "hub" = torch.hub (facebookresearch/dinov2 or dinov3 repo); "hf" = Hugging Face transformers
    backbone_source: str = "hub"
    backbone: str = "dinov2_vitb14"     # hub entry point, or HF model id / local dir when source == "hf"
    hub_repo: str = "facebookresearch/dinov2"  # or a local path to the repo for offline Kaggle runs
    hub_local: bool = False             # True -> torch.hub.load(..., source="local")
    weights_path: str = ""              # optional local .pth state_dict (offline runs)
    layer: int = -1                     # transformer block to read features from (-1 = last)
    img_size: int = 518                 # square resize; must be a multiple of patch size (14 -> 518 = 37x37)

    # ---------------- matching ----------------
    matcher: str = "soft_window"        # "argmax" | "soft_window"
    soft_window: int = 5                # window side length in (upsampled) patches
    soft_temperature: float = 0.04
    feat_upsample: int = 2              # bilinear upsampling of target feature map before matching
    flip_test: bool = False             # test-time "pose alignment": try a flipped source, keep the better one

    # ---------------- evaluation ----------------
    pck_alphas: tuple = (0.05, 0.1, 0.15)
    pck_norm: str = "bbox"              # "bbox" (SPair standard: max(w,h) of target bbox) | "img"

    # ---------------- fine-tuning (stage 2) ----------------
    ft_unfreeze_blocks: int = 2
    ft_lr: float = 1e-5
    ft_epochs: int = 2
    ft_batch_size: int = 4
    ft_temperature: float = 0.03
    ft_label_sigma: float = 1.0         # Gaussian soft-label width in patches (0 = hard one-hot)
    ft_pair_flip_aug: bool = True
    ft_max_train_pairs: int = 0         # 0 = all training pairs (cap for quick experiments)
    ft_checkpoint: str = ""             # path to load fine-tuned weights for inference

    # ---------------- misc ----------------
    device: str = "auto"
    num_workers: int = 2
    seed: int = 0
    out_dir: str = "outputs"
    run_name: str = "stage1_dinov2_b14"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls(**json.loads(Path(path).read_text()))

    def replace(self, **kw) -> "Config":
        d = asdict(self)
        d.update(kw)
        return Config(**d)
