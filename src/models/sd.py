"""Frozen Stable Diffusion features (DIFT recipe, Tang et al. 2023).

The image goes through the VAE encoder; the latent gets noise of ONE fixed step (sd_timestep, DIFT: 261)
from one fixed noise tensor, and the UNet runs once with the empty-prompt embedding. The feature map is
the output of decoder block `unet.up_blocks[sd_up_block]` (DIFT: 1), read with a forward hook; the rest
of the UNet is skipped. With `sd_up_blocks` (SD+DINO) several blocks are read: each is resized to the grid and
L2-normalised, then concatenated. With n_samples > 1 the raw features of several fixed noises are averaged
(DIFT ensemble). No denoising loop, nothing trained.
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
    """forward(x [B, 3, S, S] DINO-normalised, out_hw, n_samples) -> L2-normalised features [B, out_dim, h, w], float32."""

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

        multi = cfg.get("sd_up_blocks")
        self.up_blocks = list(multi) if multi else [cfg["sd_up_block"]]
        for block in self.up_blocks:
            if not 0 <= block < len(self.unet.up_blocks):
                raise ValueError(f"sd up-block must be in [0, {len(self.unet.up_blocks)}), got {block}")
        self.multi = bool(multi)
        self.seed = cfg.get("seed", 0)
        self.image_size = cfg["sd_image_size"]
        # Clamped for tiny test schedulers with fewer steps.
        self.timestep = min(int(cfg["sd_timestep"]), scheduler.config.num_train_timesteps - 1)
        alpha = scheduler.alphas_cumprod[self.timestep].item()
        self.noise_scale = (alpha ** 0.5, (1 - alpha) ** 0.5)
        # Stable Diffusion latent scaling factor, used when the VAE config has none
        self.latent_scale = getattr(self.vae.config, "scaling_factor", None) or 0.18215
        channels = list(reversed(self.unet.config.block_out_channels))
        self.out_dim = sum(channels[b] for b in self.up_blocks)

        vae_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if self.image_size % vae_factor:
            raise ValueError(f"sd_image_size {self.image_size} must be a multiple of the VAE factor {vae_factor}")
        latent = self.image_size // vae_factor
        self.noise_shape = (1, self.unet.config.in_channels, latent, latent)
        mean = torch.tensor(cfg["image_mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(cfg["image_std"], dtype=torch.float32).view(1, 3, 1, 1)
        # Not persistent: rebuilt from cfg, never in checkpoints.
        self.register_buffer("prompt", prompt.to(dtype), persistent=False)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        self._captured = {}
        deepest_needed = max(self.up_blocks)  # up_blocks run 0, 1, 2, ...: this one is reached last
        for block in self.up_blocks:
            self.unet.up_blocks[block].register_forward_hook(self._make_hook(block, stop=block == deepest_needed))
        self.requires_grad_(False)
        self.train(False)

    def _make_hook(self, block, stop):
        def hook(module, inputs, output):
            self._captured[block] = output
            if stop:
                raise _StopForward
        return hook

    def _noise(self, k):
        """Fixed noise of sample k: seed + k, so sample 0 is the single-sample noise."""
        g = torch.Generator().manual_seed(self.seed + k)
        return torch.randn(*self.noise_shape, generator=g)

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
    def forward(self, x, out_hw, n_samples=1):
        p = next(self.unet.parameters())
        rgb = (x.float() * self.std + self.mean) * 2 - 1
        rgb = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        latents = self.vae.encode(rgb.to(p.dtype)).latent_dist.mean * self.latent_scale
        a, b = self.noise_scale
        t = torch.full((x.shape[0],), self.timestep, device=x.device, dtype=torch.long)
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        total = {}
        for k in range(n_samples):
            # Same noise for every image: features do not depend on the batch or on source / target.
            noise = self._noise(k).to(x.device).expand(x.shape[0], -1, -1, -1)
            noisy = (a * latents.float() + b * noise).to(p.dtype)
            self._captured = {}
            try:
                self.unet(noisy, t, encoder_hidden_states=prompt)
            except _StopForward:
                pass
            for block, feats in self._captured.items():
                feats = feats.float()
                total[block] = feats if block not in total else total[block] + feats
            self._captured = {}
        outs = []
        for block in self.up_blocks:
            feats = total[block] / n_samples
            if tuple(feats.shape[-2:]) != tuple(out_hw):
                feats = F.interpolate(feats, size=tuple(out_hw), mode="bilinear", align_corners=False)
            outs.append(F.normalize(feats, dim=1) if self.multi else feats)
        return F.normalize(torch.cat(outs, dim=1), dim=1)
