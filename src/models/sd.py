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
        self.use_mid_block = bool(cfg.get("sd_mid_block", False))
        self.stop_block = max(self.up_blocks)
        self.image_size = cfg["sd_image_size"]
        # Clamped for tiny test schedulers with fewer steps.
        self.add_noise = bool(cfg.get("sd_add_noise", True))
        self.timestep = min(int(cfg["sd_timestep"]), scheduler.config.num_train_timesteps - 1)
        alpha = scheduler.alphas_cumprod[self.timestep].item()
        self.noise_scale = (alpha ** 0.5, (1 - alpha) ** 0.5)
        # Stable Diffusion latent scaling factor, used when the VAE config has none
        self.latent_scale = getattr(self.vae.config, "scaling_factor", None) or 0.18215
        rev_ch = list(reversed(self.unet.config.block_out_channels))
        mid_ch = int(self.unet.config.block_out_channels[-1])
        self.block_keys = (["mid"] if self.use_mid_block else []) + list(self.up_blocks)
        self.block_dims = ([mid_ch] if self.use_mid_block else []) + [int(rev_ch[b]) for b in self.up_blocks]
        self.out_dim = sum(self.block_dims)

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
        if self.use_mid_block:
            self.unet.mid_block.register_forward_hook(self._make_capture("mid"))
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
    def _extract_raw_blocks(self, x):
        """Run VAE + UNet once per noise draw; yield dict {block_key: float32 tensor} per draw."""
        p = next(self.unet.parameters())
        rgb = (x.float() * self.std + self.mean) * 2 - 1
        if tuple(rgb.shape[-2:]) != (self.image_size, self.image_size):
            rgb = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        latents = self.vae.encode(rgb.to(p.dtype)).latent_dist.mean * self.latent_scale
        a, b = self.noise_scale
        t = torch.full((x.shape[0],), self.timestep, device=x.device, dtype=torch.long)
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        n_draws = 1 if not self.add_noise else (self.noise_samples if self._eval_mode else 1)

        draws = []
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
            draws.append({key: self._captured[key].float() for key in self.block_keys})
            self._captured = {}
        return draws

    @torch.no_grad()
    def forward_multiscale(self, x):
        """Extract list of L2-normalised float32 block feature maps [B, C_b, H_b, W_b] at native scales."""
        draws = self._extract_raw_blocks(x)
        accs = [None] * len(self.block_keys)
        for captured in draws:
            for i, key in enumerate(self.block_keys):
                f = F.normalize(captured[key], dim=1)
                accs[i] = f if accs[i] is None else (accs[i] + f)
        return [F.normalize(f, dim=1) if len(draws) > 1 else f for f in accs]

    @torch.no_grad()
    def forward(self, x, out_hw):
        target_hw = tuple(out_hw)
        acc = None
        for captured in self._extract_raw_blocks(x):
            parts = []
            for key in self.block_keys:
                f = captured[key]
                if tuple(f.shape[-2:]) != target_hw:
                    f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
                parts.append(F.normalize(f, dim=1))
            sample_feat = F.normalize(torch.cat(parts, dim=1), dim=1) if len(parts) > 1 else parts[0]
            acc = sample_feat if acc is None else (acc + sample_feat)
        return F.normalize(acc, dim=1)


class _LayerNorm2d(nn.Module):
    """Pointwise channel LayerNorm for [B, C, H, W] (crop-size and batch-size invariant)."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels), eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _HyperBottleneck(nn.Module):
    """Pre-norm residual bottleneck [B, C, H, W] -> [B, C, H, W] with zero-init output projection."""

    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        dim, hidden = int(dim), int(hidden)
        groups = 4 if hidden % 4 == 0 else 1
        self.norm1 = _LayerNorm2d(dim)
        self.down = nn.Conv2d(dim, hidden, kernel_size=1)
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=groups)
        self.norm2 = _LayerNorm2d(hidden)
        self.drop = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.up = nn.Conv2d(hidden, dim, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = F.gelu(self.down(self.norm1(x)))
        h = F.gelu(self.norm2(self.conv(h)))
        return x + self.up(self.drop(h))


class SDHyperfeatureAdapter(nn.Module):
    """Multi-scale Diffusion Hyperfeature aggregator (Luo et al. 2023 / GeoAware-SC).

    Projects each frozen UNet block [B, C_b, H_b, W_b] into a clean `out_dim` descriptor space
    via an orthogonal-initialised 1x1 stem + native-scale residual bottleneck, aligns all scales
    to `out_hw`, fuses them with learned softmax weights, refines post-fusion, and L2-normalises.
    """

    def __init__(self, block_dims, out_dim=256, hidden=96, dropout=0.15, init_logits=None):
        super().__init__()
        if not block_dims:
            raise ValueError("block_dims must be non-empty")
        import math

        self.out_dim = int(out_dim)
        self.drop_in = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.stems = nn.ModuleList()
        self.branches = nn.ModuleList()
        for in_dim in block_dims:
            in_dim = int(in_dim)
            stem = nn.Conv2d(in_dim, self.out_dim, kernel_size=1, bias=True)
            nn.init.orthogonal_(stem.weight, gain=math.sqrt(max(in_dim / self.out_dim, 1.0)))
            nn.init.zeros_(stem.bias)
            self.stems.append(stem)
            self.branches.append(_HyperBottleneck(self.out_dim, int(hidden), dropout=dropout))

        if init_logits is not None and len(init_logits) == len(block_dims):
            logits = torch.tensor(list(init_logits), dtype=torch.float32)
        else:
            logits = torch.zeros(len(block_dims), dtype=torch.float32)
        self.mix_logits = nn.Parameter(logits)
        self.post = _HyperBottleneck(self.out_dim, int(hidden) * 2, dropout=dropout)

    def forward(self, block_feats, out_hw):
        weights = F.softmax(self.mix_logits, dim=0)
        target_hw = tuple(out_hw)
        fused = None
        for i, feat in enumerate(block_feats):
            h = self.branches[i](self.stems[i](self.drop_in(feat)))
            if tuple(h.shape[-2:]) != target_hw:
                h = F.interpolate(h, size=target_hw, mode="bilinear", align_corners=False)
            h = F.normalize(h, dim=1)
            fused = weights[i] * h if fused is None else (fused + weights[i] * h)
        return F.normalize(self.post(fused), dim=1)
