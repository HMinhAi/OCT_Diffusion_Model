"""Natural Pseudo-Anomaly Generator for Medical Images (NSA-style).

Generates realistic, irregular, soft-boundary synthetic lesions inside the lung field
to train the Autoencoder for active pathology eradication and contrastive repulsion.
References:
    - DRAEM (Zavrtanik et al., ICCV 2021)
    - Natural Synthetic Anomalies - NSA (Taher et al., MICCAI 2021)
    - CutPaste (Li et al., CVPR 2021)
"""

from __future__ import annotations

import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter


class NaturalPseudoAnomalyGenerator:
    """Generates synthetic soft infiltrates by blending source patches from other images."""

    def __init__(
        self,
        img_size: int = 224,
        margin: int = 35,  # lung parenchyma interior margin
        min_blobs: int = 1,
        max_blobs: int = 3,
        alpha_min: float = 0.20,
        alpha_max: float = 0.50,
        sigma_min: float = 3.0,
        sigma_max: float = 6.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.img_size = img_size
        self.margin = margin
        self.min_blobs = min_blobs
        self.max_blobs = max_blobs
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device

    def generate_single_mask(self) -> np.ndarray:
        """Generates an irregular, soft-edged 2D binary-to-soft mask [H, W]."""
        h = w = self.img_size
        mask = np.zeros((h, w), dtype=np.float32)
        num_blobs = random.randint(self.min_blobs, self.max_blobs)

        # Restrict blob centers to the lung field interior
        y_min, y_max = self.margin, h - self.margin
        x_min, x_max = self.margin, w - self.margin

        for _ in range(num_blobs):
            # Choose random center
            cy = random.randint(y_min, y_max)
            cx = random.randint(x_min, x_max)

            # Random radius (15 to 35 pixels)
            ry = random.randint(15, 35)
            rx = random.randint(15, 35)

            y_grid, x_grid = np.ogrid[:h, :w]
            # Elliptical distance with slight random noise
            dist = ((y_grid - cy) / ry) ** 2 + ((x_grid - cx) / rx) ** 2
            blob = (dist <= 1.0).astype(np.float32)

            # Add random texture noise to the blob boundary
            noise = np.random.uniform(0.7, 1.3, size=(h, w)).astype(np.float32)
            blob = blob * noise
            mask = np.maximum(mask, blob)

        # Smooth boundary seamlessly
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        soft_mask = gaussian_filter(mask, sigma=sigma)

        # Normalize to [0, 1]
        max_val = soft_mask.max()
        if max_val > 1e-6:
            soft_mask = soft_mask / max_val

        return soft_mask

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply pseudo-anomaly synthesis to a batch of normal images.

        Args:
            x: Batch of normal images [B, 3, H, W] in range [-1, 1].

        Returns:
            x_synth: Batch of images with synthetic lesions [B, 3, H, W].
            masks: Batch of soft anomaly masks [B, 1, H, W] in range [0, 1].
        """
        b, c, h, w = x.shape
        x_synth = x.clone()
        masks = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)

        # Shift batch to get source images from different samples
        perm = torch.randperm(b)
        x_src = x[perm]

        for i in range(b):
            mask_np = self.generate_single_mask()
            mask_t = torch.from_numpy(mask_np).to(device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, H, W]

            # Opacity factor alpha (simulating ground-glass / alveolar opacity)
            alpha = random.uniform(self.alpha_min, self.alpha_max)

            # Intensity blend: blend source image patch or intensity shift
            # In chest X-rays, pneumonia appears as radiopaque (brighter) hazy opacities:
            # We can use a mixture of source patch blending AND positive intensity shift (+0.30 to +0.60)
            use_bright_shift = random.random() < 0.5
            if use_bright_shift:
                bright_patch = (x[i] + random.uniform(0.35, 0.70)).clamp(-1.0, 1.0)
                lesion_src = 0.5 * x_src[i] + 0.5 * bright_patch
            else:
                lesion_src = x_src[i]

            x_synth[i] = (1.0 - alpha * mask_t) * x[i] + (alpha * mask_t) * lesion_src
            masks[i, 0] = mask_t.squeeze(0)

        x_synth = x_synth.clamp(-1.0, 1.0)
        return x_synth, masks
