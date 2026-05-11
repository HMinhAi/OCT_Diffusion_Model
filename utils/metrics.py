from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def min_max_normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a tensor to [0, 1] per-sample."""
    b = tensor.shape[0]
    flat = tensor.view(b, -1)
    min_vals = flat.min(dim=1, keepdim=True).values
    max_vals = flat.max(dim=1, keepdim=True).values
    normalized = (flat - min_vals) / (max_vals - min_vals + eps)
    return normalized.view_as(tensor)


def anomaly_map_to_image_score(
    anomaly_map: torch.Tensor,
    reduction: str = "max",
) -> torch.Tensor:
    """Reduce [B, 1, H, W] maps to image-level anomaly scores."""
    flat = anomaly_map.view(anomaly_map.shape[0], -1)
    if reduction == "mean":
        return flat.mean(dim=1)
    if reduction == "sum":
        return flat.sum(dim=1)
    return flat.max(dim=1).values


def safe_auroc(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # 1. Kiểm tra nếu chỉ có 1 lớp (toàn 0 hoặc toàn 1)
    if len(np.unique(y_true)) < 2:
        print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
        return 0.5 # Trả về mức ngẫu nhiên nếu không đủ dữ liệu
    
    # 2. Xử lý NaN hoặc giá trị vô cùng trong scores (nếu có)
    y_score = np.nan_to_num(y_score)
    
    try:
        # Ép kiểu rõ ràng cho bài toán nhị phân
        return float(roc_auc_score(y_true, y_score))
    except Exception as e:
        print(f"Error computing AUROC: {e}")
        return 0.5


def compute_image_auroc(labels: Iterable[int], scores: Iterable[float]) -> float:
    return safe_auroc(np.array(list(labels)), np.array(list(scores)))


def compute_pixel_auroc(
    anomaly_maps: List[np.ndarray],
    masks: List[np.ndarray],
) -> float:
    """Compute pixel-level AUROC from aligned map/mask lists."""
    if len(anomaly_maps) == 0 or len(masks) == 0:
        return float("nan")

    map_flat = np.concatenate([m.reshape(-1) for m in anomaly_maps], axis=0)
    mask_flat = np.concatenate([m.reshape(-1) for m in masks], axis=0)
    mask_flat = (mask_flat > 0).astype(np.int64)

    return safe_auroc(mask_flat, map_flat)


def extract_pixel_lists(
    batch_maps: torch.Tensor,
    batch_masks: torch.Tensor,
    batch_has_mask: torch.Tensor,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Gather map/mask arrays only for samples that have GT masks."""
    maps_out: List[np.ndarray] = []
    masks_out: List[np.ndarray] = []

    for i in range(batch_maps.shape[0]):
        if bool(batch_has_mask[i].item()):
            maps_out.append(batch_maps[i, 0].detach().cpu().numpy())
            masks_out.append(batch_masks[i, 0].detach().cpu().numpy())

    return maps_out, masks_out


def summarize_metrics(image_auc: float, pixel_auc: float, pixel_available: bool) -> Dict[str, float]:
    output: Dict[str, float] = {"image_auroc": image_auc}
    if pixel_available:
        output["pixel_auroc"] = pixel_auc
    else:
        output["pixel_auroc"] = float("nan")
    return output
