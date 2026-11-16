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


def local_sd_dir(cfg):
    """Folder with a local snapshot of sd_model_id, e.g. weights/stable-diffusion-2-1."""
    return Path(cfg["weights_dir"]) / cfg["sd_model_id"].split("/")[-1]


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

        local_dir = local_sd_dir(cfg)
        source, revision = (local_dir, None) if local_dir.is_dir() else (cfg["sd_model_id"], cfg["sd_model_revision"])
        # The dtype is chosen here, from the device build_model will move the model to: loading fp16
        # directly avoids a transient fp32 copy of the UNet on the GPU. _apply() fixes it if the
        # module is moved to another device type later.
        dtype = torch.float16 if str(resolved_device(cfg)).startswith("cuda") else torch.float32
        self.unet = _load(UNet2DConditionModel, source, revision, "unet", dtype)
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

        self.up_block = cfg["sd_up_block"]
        if not 0 <= self.up_block < len(self.unet.up_blocks):
            raise ValueError(f"sd_up_block must be in [0, {len(self.unet.up_blocks)}), got {self.up_block}")
        self.image_size = cfg["sd_image_size"]
        # Clamped for tiny test schedulers with fewer steps.
        self.timestep = min(int(cfg["sd_timestep"]), scheduler.config.num_train_timesteps - 1)
        alpha = scheduler.alphas_cumprod[self.timestep].item()
        self.noise_scale = (alpha ** 0.5, (1 - alpha) ** 0.5)
        # Stable Diffusion latent scaling factor, used when the VAE config has none
        self.latent_scale = getattr(self.vae.config, "scaling_factor", None) or 0.18215
        self.out_dim = list(reversed(self.unet.config.block_out_channels))[self.up_block]

        vae_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if self.image_size % vae_factor:
            raise ValueError(f"sd_image_size {self.image_size} must be a multiple of the VAE factor {vae_factor}")
        latent = self.image_size // vae_factor
        g = torch.Generator().manual_seed(cfg.get("seed", 0))
        noise = torch.randn(1, self.unet.config.in_channels, latent, latent, generator=g)
        mean = torch.tensor(cfg["image_mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(cfg["image_std"], dtype=torch.float32).view(1, 3, 1, 1)
        # Not persistent: rebuilt from cfg, never in checkpoints.
        self.register_buffer("prompt", prompt.to(dtype), persistent=False)
        self.register_buffer("noise", noise, persistent=False)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        self._captured = None
        self.unet.up_blocks[self.up_block].register_forward_hook(self._capture)
        self.requires_grad_(False)
        self.train(False)

    def _capture(self, module, inputs, output):
        self._captured = output
        raise _StopForward

    def train(self, mode=True):
        """Always in eval mode, whatever the parent model does."""
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
        # Same noise for every image: features do not depend on the batch or on source / target.
        noise = self.noise.expand(x.shape[0], -1, -1, -1)
        a, b = self.noise_scale
        noisy = (a * latents.float() + b * noise).to(p.dtype)
        t = torch.full((x.shape[0],), self.timestep, device=x.device, dtype=torch.long)
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        self._captured = None
        try:
            self.unet(noisy, t, encoder_hidden_states=prompt)
        except _StopForward:
            pass
        feats, self._captured = self._captured.float(), None
        if tuple(feats.shape[-2:]) != tuple(out_hw):
            feats = F.interpolate(feats, size=tuple(out_hw), mode="bilinear", align_corners=False)
        return F.normalize(feats, dim=1)
