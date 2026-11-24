"""Frozen Stable Diffusion features (DIFT recipe, Tang et al. 2023).

The image goes through the VAE encoder; the latent gets noise of ONE fixed step (sd_timestep, DIFT: 261)
from one fixed noise tensor, and the UNet runs once with the empty-prompt embedding. The feature map is
the output of decoder block `unet.up_blocks[sd_up_block]` (DIFT: 1), read with a forward hook; the rest
of the UNet is skipped. No denoising loop, nothing trained.
Grid: SD 2.1 at sd_image_size 736 -> latent 92 (VAE /8) -> up-block 1 at 1/16 of the input = 46 cells,
the same grid as DINOv2 at 644 / 14. Other sizes are bilinearly resized to the DINOv2 grid.
diffusers is imported here only, and this module only when fusion != "none".
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import resolved_device


class _StopForward(Exception):
    """Raised by the up-block hook: everything after that block is not needed."""


SD_COMPLETE_MARKER = ".download_complete"  # written by scripts/download_weights.py after the last file


def local_sd_dir(cfg):
    """Folder with a local snapshot of sd_model_id, e.g. weights/stable-diffusion-2-1."""
    return Path(cfg["weights_dir"]) / cfg["sd_model_id"].split("/")[-1]


def local_sd_ready(cfg):
    """True when local_sd_dir holds a finished download of sd_model_revision (not one cut by a disconnect)."""
    marker = local_sd_dir(cfg) / SD_COMPLETE_MARKER
    return marker.is_file() and marker.read_text().strip() == cfg["sd_model_revision"]


def _load(cls, source, revision, subfolder, dtype):
    """from_pretrained preferring the fp16 safetensors variant, then any safetensors, then .bin
    (the tiny test repo has only .bin files and no variant)."""
    attempts = ({"variant": "fp16", "use_safetensors": True}, {"use_safetensors": True}, {})
    for extra in attempts[:-1]:
        try:
            return cls.from_pretrained(source, subfolder=subfolder, revision=revision, torch_dtype=dtype, **extra)
        except (OSError, ValueError):
            pass
    return cls.from_pretrained(source, subfolder=subfolder, revision=revision, torch_dtype=dtype, **attempts[-1])


class SDFeatures(nn.Module):
    """forward(x [B, 3, S, S] DINO-normalised, out_hw) -> L2-normalised features [B, out_dim, h, w], float32."""

    def __init__(self, cfg):
        super().__init__()
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        if local_sd_ready(cfg):
            source, revision = local_sd_dir(cfg), None
        else:
            source, revision = cfg["sd_model_id"], cfg["sd_model_revision"]
            print(f"Stable Diffusion: no finished download in {local_sd_dir(cfg)}; loading {source} from the Hub "
                  "(Colab: run scripts/download_weights.py once to keep it on Drive)")
        # The dtype is chosen here, from the device build_model will move the model to: loading fp16
        # directly avoids a transient fp32 copy of the UNet on the GPU. _apply() fixes it if the
        # module is moved to another device type later.
        dtype = torch.float16 if str(resolved_device(cfg)).startswith("cuda") else torch.float32
        self.unet = _load(UNet2DConditionModel, source, revision, "unet", dtype)
        if cfg.get("sd_cleandift_repo") and cfg.get("sd_cleandift_file"):
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            cd_dir = Path(cfg["weights_dir"]) / "cleandift"
            local_cd = cd_dir / cfg["sd_cleandift_file"]
            if local_cd.is_file():
                ckpt_pth = str(local_cd)
            else:
                cd_dir.mkdir(parents=True, exist_ok=True)
                ckpt_pth = hf_hub_download(repo_id=cfg["sd_cleandift_repo"],
                                           filename=cfg["sd_cleandift_file"],
                                           local_dir=str(cd_dir))
            state_dict = load_file(ckpt_pth)
            self.unet.load_state_dict(state_dict, strict=True)
            self.unet.to(dtype)
            print(f"Loaded CleanDIFT UNet weights from {ckpt_pth}")
        self.vae = _load(AutoencoderKL, source, revision, "vae", dtype)
        scheduler = DDPMScheduler.from_config(DDPMScheduler.load_config(source, subfolder="scheduler",
                                                                        revision=revision))

        tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer", revision=revision)
        text_encoder = CLIPTextModel.from_pretrained(source, subfolder="text_encoder", revision=revision)
        tokens = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
        with torch.no_grad():
            prompt = text_encoder(tokens.input_ids)[0]  # [1, 77, D]
        del text_encoder, tokenizer

        raw_blocks = cfg.get("sd_up_blocks")
        self.up_blocks = sorted({int(b) for b in raw_blocks}) if raw_blocks else [int(cfg["sd_up_block"])]
        for b in self.up_blocks:
            if not 0 <= b < len(self.unet.up_blocks):
                raise ValueError(f"sd_up_block must be in [0, {len(self.unet.up_blocks)}), got {b}")
        self.stop_block = max(self.up_blocks)
        self.image_size = cfg["sd_image_size"]
        # Clamped for tiny test schedulers with fewer steps.
        self.add_noise = bool(cfg.get("sd_add_noise", True))
        self.timestep = 0 if not self.add_noise else min(int(cfg["sd_timestep"]), scheduler.config.num_train_timesteps - 1)
        alpha = scheduler.alphas_cumprod[self.timestep].item()
        self.noise_scale = (alpha ** 0.5, (1 - alpha) ** 0.5)
        # Stable Diffusion latent scaling factor, used when the VAE config has none
        self.latent_scale = getattr(self.vae.config, "scaling_factor", None) or 0.18215
        rev_ch = list(reversed(self.unet.config.block_out_channels))
        self.out_dim = sum(rev_ch[b] for b in self.up_blocks)

        vae_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if self.image_size % vae_factor:
            raise ValueError(f"sd_image_size {self.image_size} must be a multiple of the VAE factor {vae_factor}")
        latent = self.image_size // vae_factor
        self.noise_samples = max(1, int(cfg.get("sd_noise_samples", 1)))
        g = torch.Generator().manual_seed(cfg.get("seed", 0))
        noise = torch.randn(self.noise_samples, self.unet.config.in_channels, latent, latent, generator=g)
        mean = torch.tensor(cfg["image_mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(cfg["image_std"], dtype=torch.float32).view(1, 3, 1, 1)
        # Not persistent: rebuilt from cfg, never in checkpoints.
        self.register_buffer("prompt", prompt.to(dtype), persistent=False)
        self.register_buffer("noise", noise, persistent=False)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        self._captured = {}
        self._eval_mode = True
        for b in self.up_blocks:
            self.unet.up_blocks[b].register_forward_hook(self._make_capture(b))
        self.requires_grad_(False)
        self.train(False)

    def _make_capture(self, idx):
        def _hook(module, inputs, output):
            self._captured[idx] = output
            if idx == self.stop_block:
                raise _StopForward
        return _hook

    def train(self, mode=True):
        """UNet/VAE always stay in eval mode; track parent mode to use 1 noise sample in train vs K in eval."""
        self._eval_mode = not bool(mode)
        return super().train(False)

    def _apply(self, fn, *args, **kwargs):
        """After a move: fp16 weights on CUDA, fp32 elsewhere (fp16 convolutions are slow or missing on CPU).
        mean / std / noise stay fp32: they are cast at forward."""
        super()._apply(fn, *args, **kwargs)
        p = next(self.unet.parameters())
        want = torch.float16 if p.device.type == "cuda" else torch.float32
        if p.dtype != want:
            for module in (self.unet, self.vae):
                module.to(want)
            self.prompt = self.prompt.to(want)
        return self

    @torch.no_grad()
    def forward(self, x, out_hw):
        p = next(self.unet.parameters())
        rgb = (x.float() * self.std + self.mean) * 2 - 1
        rgb = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        latents = self.vae.encode(rgb.to(p.dtype)).latent_dist.mean * self.latent_scale
        a, b = self.noise_scale
        t = torch.full((x.shape[0],), self.timestep, device=x.device, dtype=torch.long)
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        n_draws = 1 if not self.add_noise else (self.noise_samples if self._eval_mode else 1)

        acc = None
        for k in range(n_draws):
            if self.add_noise:
                noise = self.noise[k : k + 1].expand(x.shape[0], -1, -1, -1)
                noisy = (a * latents.float() + b * noise).to(p.dtype)
            else:
                noisy = latents.to(p.dtype)
            self._captured = {}
            try:
                self.unet(noisy, t, encoder_hidden_states=prompt)
            except _StopForward:
                pass
            parts = []
            for blk in self.up_blocks:
                f = self._captured[blk].float()
                if tuple(f.shape[-2:]) != tuple(out_hw):
                    f = F.interpolate(f, size=tuple(out_hw), mode="bilinear", align_corners=False)
                parts.append(F.normalize(f, dim=1))
            self._captured = {}
            sample_feat = F.normalize(torch.cat(parts, dim=1), dim=1) if len(parts) > 1 else parts[0]
            acc = sample_feat if acc is None else (acc + sample_feat)
        return F.normalize(acc, dim=1)


class SDCorrespondenceAdapter(nn.Module):
    """Lightweight zero-init residual bottleneck adapter on multi-block SD features [B, C, h, w].

    At step 0, `out` is zero so output == frozen multi-block DIFT features exactly.
    With in_dim=3200, hidden=128, layers=2: ~1.41M trainable parameters (ideal for 314 pairs).
    """

    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.drop = nn.Dropout2d(float(dropout))
        self.inp = nn.Conv2d(in_dim, hidden, kernel_size=1)
        groups = 8 if hidden % 8 == 0 else 1
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(groups, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(groups, hidden),
            )
            for _ in range(int(layers))
        ])
        self.out = nn.Conv2d(hidden, in_dim, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = F.gelu(self.inp(self.drop(x)))
        for blk in self.blocks:
            h = F.gelu(h + blk(h))
        return F.normalize(x + self.out(h), dim=1)
