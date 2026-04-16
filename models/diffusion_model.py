from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm_groups(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = timestep.device
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = timestep.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetDenoiser(nn.Module):
    """U-Net denoiser for epsilon prediction in DDPM."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        time_dim = model_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.skip_channels = []
        current_channels = model_channels

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = ResidualBlock(current_channels, out_ch, time_dim, dropout=dropout)
                self.down_blocks.append(block)
                current_channels = out_ch
                self.skip_channels.append(current_channels)
            if level != len(channel_mult) - 1:
                down = Downsample(current_channels)
                self.down_blocks.append(down)
                self.skip_channels.append(current_channels)

        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_dim, dropout=dropout)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        skip_channels = list(self.skip_channels)

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                block = ResidualBlock(current_channels + skip_ch, out_ch, time_dim, dropout=dropout)
                self.up_blocks.append(block)
                current_channels = out_ch
            if level != 0:
                skip_ch = skip_channels.pop()
                block = ResidualBlock(current_channels + skip_ch, out_ch, time_dim, dropout=dropout)
                self.up_blocks.append(block)
                self.up_blocks.append(Upsample(out_ch))
                current_channels = out_ch

        self.output_norm = nn.GroupNorm(_group_norm_groups(current_channels), current_channels)
        self.output_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h = self.input_conv(x)
        skips = []

        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)

        return self.output_conv(F.silu(self.output_norm(h)))


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model wrapper."""

    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    @staticmethod
    def _extract(coeff: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        out = coeff.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_t * x0 + sqrt_one_minus_t * noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x_t, t)

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.predict_noise(x_t, t)
        return F.mse_loss(noise_pred, noise)

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps_theta = self.predict_noise(x_t, t)

        beta_t = self._extract(self.betas, t, x_t.shape)
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)

        sqrt_recip_alpha_t = torch.rsqrt(alpha_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / (sqrt_one_minus_alpha_cumprod_t + 1e-8)) * eps_theta
        )
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        return model_mean, posterior_var, eps_theta

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_mean, posterior_var, eps_theta = self.p_mean_variance(x_t, t)

        if deterministic:
            noise = torch.zeros_like(x_t)
        else:
            noise = torch.randn_like(x_t)

        nonzero_mask = (t > 0).float().view(t.shape[0], 1, 1, 1)
        sample = model_mean + nonzero_mask * torch.sqrt(posterior_var) * noise
        return sample, eps_theta

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_size: int,
        channels: int,
        device: torch.device,
    ) -> torch.Tensor:
        x = torch.randn((batch_size, channels, image_size, image_size), device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, t, deterministic=False)
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def reconstruct_via_inversion(
        self,
        x0: torch.Tensor,
        inversion_steps: int,
        deterministic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward diffuse to t, then reverse to t=0 for inversion reconstruction."""
        batch_size = x0.shape[0]
        t_val = min(max(int(inversion_steps), 1), self.timesteps - 1)
        t = torch.full((batch_size,), t_val, device=x0.device, dtype=torch.long)

        true_noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, true_noise)
        pred_noise_t = self.predict_noise(x_t, t)

        current = x_t
        for step in reversed(range(t_val + 1)):
            t_step = torch.full((batch_size,), step, device=x0.device, dtype=torch.long)
            current, _ = self.p_sample(current, t_step, deterministic=deterministic)

        return {
            "x_recon": current.clamp(-1.0, 1.0),
            "x_t": x_t,
            "true_noise": true_noise,
            "pred_noise": pred_noise_t,
            "timestep": t,
        }


def build_ddpm_from_config(config: Dict) -> DDPM:
    diff_cfg = config["diffusion"]

    denoiser = UNetDenoiser(
        in_channels=3,
        out_channels=3,
        model_channels=diff_cfg.get("model_channels", 64),
        channel_mult=tuple(diff_cfg.get("channel_mult", [1, 2, 4, 8])),
        num_res_blocks=diff_cfg.get("num_res_blocks", 2),
        dropout=diff_cfg.get("dropout", 0.0),
    )

    ddpm = DDPM(
        denoiser=denoiser,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
    )
    return ddpm
