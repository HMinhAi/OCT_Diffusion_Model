from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_to_numpy_image(
    image_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Convert normalized CHW tensor to RGB HWC numpy image in [0, 1]."""
    image = image_tensor.detach().cpu().float().clone()
    for i in range(3):
        image[i] = image[i] * float(std[i]) + float(mean[i])
    image = torch.clamp(image, 0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def normalize_map(anomaly_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32)
    min_v = float(np.min(anomaly_map))
    max_v = float(np.max(anomaly_map))
    return (anomaly_map - min_v) / (max_v - min_v + eps)


def overlay_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.45,
    cmap_name: str = "jet",
    normalize: bool = True,
) -> np.ndarray:
    """Overlay a heatmap onto an RGB image."""
    heat = normalize_map(anomaly_map) if normalize else np.clip(anomaly_map.astype(np.float32), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(heat)[..., :3]
    blended = (1.0 - alpha) * image + alpha * heat_rgb
    return np.clip(blended, 0.0, 1.0)


def save_anomaly_panel(
    save_path: str,
    image_tensor: torch.Tensor,
    diff_map: torch.Tensor,
    feat_map: torch.Tensor,
    fused_map: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    noise_map: Optional[torch.Tensor] = None,
    attention_maps: Optional[dict[str, torch.Tensor]] = None,
    title: Optional[str] = None,
    normalize_maps: bool = True,
) -> None:
    """Save side-by-side visualization panel for anomaly maps."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_np = tensor_to_numpy_image(image_tensor, mean=mean, std=std)
    diff_np = diff_map.detach().cpu().squeeze().numpy()
    feat_np = feat_map.detach().cpu().squeeze().numpy()
    fused_np = fused_map.detach().cpu().squeeze().numpy()

    if noise_map is not None:
        noise_np = noise_map.detach().cpu().squeeze().numpy()

    attention_items = []
    if attention_maps is not None:
        attention_items = [
            ("Alpha Diff", attention_maps.get("alpha")),
            ("Beta Feat", attention_maps.get("beta")),
            ("Gamma Noise", attention_maps.get("gamma")),
        ]
        attention_items = [(name, tensor) for name, tensor in attention_items if tensor is not None]

    n_cols = 5 if noise_map is not None else 4
    n_cols += len(attention_items)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.0))

    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(overlay_heatmap(image_np, diff_np, normalize=normalize_maps))
    axes[1].set_title("Diffusion Map")
    axes[1].axis("off")

    axes[2].imshow(overlay_heatmap(image_np, feat_np, normalize=normalize_maps))
    axes[2].set_title("Feature Map")
    axes[2].axis("off")

    col_offset = 3
    if noise_map is not None:
        axes[3].imshow(overlay_heatmap(image_np, noise_np, normalize=normalize_maps))
        axes[3].set_title("Simplex Noise Map")
        axes[3].axis("off")
        col_offset = 4

    axes[col_offset].imshow(overlay_heatmap(image_np, fused_np, normalize=normalize_maps))
    axes[col_offset].set_title("Final Fused Map")
    axes[col_offset].axis("off")

    for offset, (name, tensor) in enumerate(attention_items, start=col_offset + 1):
        attn_np = tensor.detach().cpu().squeeze().numpy()
        axes[offset].imshow(overlay_heatmap(image_np, attn_np, alpha=0.55, normalize=True))
        axes[offset].set_title(name)
        axes[offset].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
