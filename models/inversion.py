from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from models.diffusion_model import DDPM


@dataclass
class InversionOutput:
    x_recon: torch.Tensor
    diffusion_map: torch.Tensor
    noise_map: torch.Tensor
    combined_map: torch.Tensor


class SimplexNoise2D:
    """Minimal 2D simplex noise generator for robustness injection."""

    _GRAD3 = np.array(
        [
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
        ],
        dtype=np.float32,
    )

    _F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    _G2 = (3.0 - math.sqrt(3.0)) / 6.0

    def __init__(self, seed: int = 0) -> None:
        permutation = list(range(256))
        rnd = random.Random(seed)
        rnd.shuffle(permutation)
        self.perm = permutation * 2

    def _dot(self, g: np.ndarray, x: float, y: float) -> float:
        return float(g[0] * x + g[1] * y)

    def noise2d(self, xin: float, yin: float) -> float:
        s = (xin + yin) * self._F2
        i = math.floor(xin + s)
        j = math.floor(yin + s)
        t = (i + j) * self._G2
        x0 = xin - (i - t)
        y0 = yin - (j - t)

        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1

        x1 = x0 - i1 + self._G2
        y1 = y0 - j1 + self._G2
        x2 = x0 - 1.0 + 2.0 * self._G2
        y2 = y0 - 1.0 + 2.0 * self._G2

        ii = i & 255
        jj = j & 255
        gi0 = self.perm[ii + self.perm[jj]] % 12
        gi1 = self.perm[ii + i1 + self.perm[jj + j1]] % 12
        gi2 = self.perm[ii + 1 + self.perm[jj + 1]] % 12

        n0 = 0.0
        n1 = 0.0
        n2 = 0.0

        t0 = 0.5 - x0 * x0 - y0 * y0
        if t0 >= 0:
            t0 *= t0
            n0 = t0 * t0 * self._dot(self._GRAD3[gi0], x0, y0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        if t1 >= 0:
            t1 *= t1
            n1 = t1 * t1 * self._dot(self._GRAD3[gi1], x1, y1)

        t2 = 0.5 - x2 * x2 - y2 * y2
        if t2 >= 0:
            t2 *= t2
            n2 = t2 * t2 * self._dot(self._GRAD3[gi2], x2, y2)

        return 70.0 * (n0 + n1 + n2)

    def generate(
        self,
        height: int,
        width: int,
        scale: float = 0.05,
        octaves: int = 3,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> np.ndarray:
        noise = np.zeros((height, width), dtype=np.float32)
        frequency = scale
        amplitude = 1.0
        amplitude_sum = 0.0

        for _ in range(max(octaves, 1)):
            for y in range(height):
                yy = y * frequency
                for x in range(width):
                    xx = x * frequency
                    noise[y, x] += self.noise2d(xx, yy) * amplitude
            amplitude_sum += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        if amplitude_sum > 0:
            noise /= amplitude_sum

        min_v = noise.min()
        max_v = noise.max()
        if max_v - min_v > 1e-8:
            noise = 2.0 * (noise - min_v) / (max_v - min_v) - 1.0

        return noise


def generate_batch_simplex_noise(
    batch_size: int,
    height: int,
    width: int,
    scale: float,
    octaves: int,
    base_seed: int,
    device: torch.device,
) -> torch.Tensor:
    maps = []
    for idx in range(batch_size):
        generator = SimplexNoise2D(seed=base_seed + idx)
        map_np = generator.generate(height=height, width=width, scale=scale, octaves=octaves)
        maps.append(torch.from_numpy(map_np).float().unsqueeze(0))
    return torch.stack(maps, dim=0).to(device=device)


class DiffusionInversionDetector:
    """Inference helper for inversion-based anomaly detection."""

    def __init__(
        self,
        ddpm: DDPM,
        anomaly_distance: str = "l1",
        inversion_steps: int = 50,
        use_noise_space_error: bool = True,
    ) -> None:
        self.ddpm = ddpm
        self.anomaly_distance = anomaly_distance.lower()
        self.inversion_steps = inversion_steps
        self.use_noise_space_error = use_noise_space_error

    @torch.no_grad()
    def invert(self, x: torch.Tensor) -> InversionOutput:
        inv = self.ddpm.reconstruct_via_inversion(
            x0=x,
            inversion_steps=self.inversion_steps,
            deterministic=True,
        )
        x_recon = inv["x_recon"]
        true_noise = inv["true_noise"]
        pred_noise = inv["pred_noise"]

        if self.anomaly_distance == "l2":
            diffusion_map = ((x - x_recon) ** 2).mean(dim=1, keepdim=True)
        else:
            diffusion_map = (x - x_recon).abs().mean(dim=1, keepdim=True)

        noise_map = (pred_noise - true_noise).abs().mean(dim=1, keepdim=True)

        if self.use_noise_space_error:
            combined_map = diffusion_map + noise_map
        else:
            combined_map = diffusion_map

        return InversionOutput(
            x_recon=x_recon,
            diffusion_map=diffusion_map,
            noise_map=noise_map,
            combined_map=combined_map,
        )

    @torch.no_grad()
    def simplex_robust_anomaly(
        self,
        x: torch.Tensor,
        runs: int,
        noise_scale: float,
        octaves: int,
        aggregate: str = "mean",
    ) -> torch.Tensor:
        """Aggregate anomaly map over multiple simplex-noisy forward passes."""
        batch_size, channels, height, width = x.shape
        maps = []

        for run_id in range(max(runs, 1)):
            simplex = generate_batch_simplex_noise(
                batch_size=batch_size,
                height=height,
                width=width,
                scale=0.025,
                octaves=octaves,
                base_seed=17 * (run_id + 1),
                device=x.device,
            )
            simplex = simplex.repeat(1, channels, 1, 1)
            x_noisy = torch.clamp(x + noise_scale * simplex, min=-1.0, max=1.0)
            maps.append(self.invert(x_noisy).combined_map)

        stacked = torch.stack(maps, dim=0)
        if aggregate.lower() in {"var", "variance"}:
            return stacked.var(dim=0, unbiased=False)
        return stacked.mean(dim=0)


def apply_deviation_correction(
    diffusion_map: torch.Tensor,
    feature_map: torch.Tensor,
    lambda_correction: float,
) -> torch.Tensor:
    """Deviation-corrected anomaly map: S_corr = S_diffusion - lambda * S_feature."""
    corrected = diffusion_map - lambda_correction * feature_map
    return torch.clamp(corrected, min=0.0)
