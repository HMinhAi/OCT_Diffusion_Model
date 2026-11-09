"""Anomaly Scoring module for Decoupled Dual-Stream PatchCL-AE.

Reference:
    Section 3.2: Testing process and anomaly score
    Section 3.2.1: Patch-wise anomaly score

Inference Innovation:
    1. Cosine Discrepancy Map:
           M_l(h, w) = 1 - cos(P_l(E_l(X))(h, w), P_l(E_l(X_hat))(h, w))
    2. Global Baseline Thresholding (tau_l^{global}):
           Ngưỡng tau_l được tính cố định 1 lần duy nhất trên toàn tập Validation Normal:
               tau_l^{global} = Quantile_{0.95}( \bigcup_{X \in D_{val_norm}} M_l(X) )
           Với ảnh Normal X_norm: M_l(X_norm) < tau_l^{global}
           Kết quả: \tilde{M}_l = ReLU( M_l - tau_l^{global} ) = 0 -> A(X_norm) \approx 0.0
    3. Multi-scale Deep-Layer Fusion:
           M_{final} = 0.05 * \tilde{M}_1 + 0.15 * \tilde{M}_2 + 0.35 * \tilde{M}_3 + 0.45 * \tilde{M}_4
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_layer_similarity_map(
    feat_real: torch.Tensor,
    feat_recon: torch.Tensor,
    projection_head: Optional[nn.Module] = None,
    use_projection_head: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute local semantic discrepancy map M_l for a single layer l.
    
    Khi Evaluation / Inference: KHÔNG DÙNG Projection Head P_l, mà tính Cosine Distance
    trực tiếp trên đặc trưng nguyên bản của Encoder E_l:
        f_real = encoder_features[l](X)        # Shape: [B, C, H, W]
        f_rec  = encoder_features[l](X_hat)    # Shape: [B, C, H, W]
        cos_sim = F.cosine_similarity(f_real, f_rec, dim=1) # Shape: [B, H, W]
        M_l = 1.0 - cos_sim                    # Shape: [B, 1, H, W]
        
    Args:
        feat_real: Feature map nguyên bản từ input image X [B, C, H, W]
        feat_recon: Feature map nguyên bản từ reconstructed image X_hat [B, C, H, W]
        projection_head: (Optional) ProjectionHead P_l (mặc định không sử dụng khi inference)
        use_projection_head: bool (Mặc định False: không dùng P_l để tránh lỗi trực giao vector)
        eps: Small epsilon for numerical stability
        
    Returns:
        M_l: Anomaly score map [B, 1, H, W]
    """
    if not use_projection_head or projection_head is None:
        # Tính Cosine Similarity trực tiếp dọc theo chiều Channel (dim=1)
        cos_sim = F.cosine_similarity(feat_real, feat_recon, dim=1, eps=eps)  # [B, H, W]
        return (1.0 - cos_sim).unsqueeze(1)  # [B, 1, H, W]
    else:
        # Fallback cũ nếu use_projection_head=True
        b, c, h, w = feat_real.shape
        real_vec = feat_real.permute(0, 2, 3, 1).contiguous().view(-1, c)
        recon_vec = feat_recon.permute(0, 2, 3, 1).contiguous().view(-1, c)

        f_real = projection_head(real_vec).view(b, h, w, -1)     # [B, H, W, 256]
        f_recon = projection_head(recon_vec).view(b, h, w, -1)   # [B, H, W, 256]

        f_real_norm = F.normalize(f_real, p=2, dim=-1, eps=eps)
        f_recon_norm = F.normalize(f_recon, p=2, dim=-1, eps=eps)

        cos_sim = (f_real_norm * f_recon_norm).sum(dim=-1, keepdim=True)  # [B, H, W, 1]
        anomaly_map_l = 1.0 - cos_sim
        return anomaly_map_l.permute(0, 3, 1, 2)  # [B, 1, H, W]


def compute_anomaly_score_simple(
    layer_maps: Union[List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    weights: Union[List[float], Tuple[float, ...]] = (0.05, 0.15, 0.35, 0.45),
    top_k_ratio: float = 0.10,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    """Quy trình tính Anomaly Score chuẩn sau khi BỎ Soft-Thresholding.
    
    1. Cộng dồn trực tiếp 4 tầng theo trọng số lệch tầng sâu:
           combined_map = weights[0]*M_1 + weights[1]*M_2 + weights[2]*M_3 + weights[3]*M_4
    2. Lấy trung bình Top 10% pixel cao nhất của map tổng làm điểm bức ảnh.
    
    Args:
        layer_maps: list 4 tensor [M_E1, M_E2, M_E3, M_E4] với M_l = 1 - cos_sim,
                    hoặc dict {'E_1': M_1, 'E_2': M_2, 'E_3': M_3, 'E_4': M_4}.
        weights: Trọng số lệch tầng sâu (mặc định: [0.05, 0.15, 0.35, 0.45]).
        top_k_ratio: Tỷ lệ Top pixel cao nhất lấy trung bình (mặc định: 0.10 tức 10%).
        target_size: Kích thước không gian mục tiêu (H, W) để nội suy nếu các tầng khác độ phân giải.
        
    Returns:
        combined_map: Bản đồ tổng cộng dồn 4 tầng [B, 1, H, W] hoặc [H, W]
        img_score: Điểm bức ảnh (float nếu đơn lẻ/B=1, hoặc 1D Tensor [B] nếu batch > 1).
    """
    if isinstance(layer_maps, dict):
        maps_list = [layer_maps[k] for k in ["E_1", "E_2", "E_3", "E_4"] if k in layer_maps]
    else:
        maps_list = list(layer_maps)

    if len(maps_list) != len(weights):
        raise ValueError(f"Số lượng layer_maps ({len(maps_list)}) phải khớp với weights ({len(weights)})")

    # Xác định target_size nếu chưa truyền
    if target_size is None:
        target_size = (max(m.shape[-2] for m in maps_list), max(m.shape[-1] for m in maps_list))

    # Đảm bảo các tầng có cùng kích thước không gian target_size trước khi cộng dồn
    interpolated_maps = []
    for m in maps_list:
        if m.shape[-2:] != target_size:
            if m.dim() == 2:
                m_4d = m.unsqueeze(0).unsqueeze(0)
                m_res = F.interpolate(m_4d, size=target_size, mode="bilinear", align_corners=True).squeeze(0).squeeze(0)
            elif m.dim() == 3:
                m_4d = m.unsqueeze(1)
                m_res = F.interpolate(m_4d, size=target_size, mode="bilinear", align_corners=True).squeeze(1)
            else:
                m_res = F.interpolate(m, size=target_size, mode="bilinear", align_corners=True)
            interpolated_maps.append(m_res)
        else:
            interpolated_maps.append(m)

    # 1. Cộng dồn các tầng theo trọng số
    combined_map = torch.zeros_like(interpolated_maps[0])
    for w, m in zip(weights, interpolated_maps):
        combined_map = combined_map + w * m

    # 2. Lấy trung bình Top 10% pixel cao nhất của map tổng làm điểm bức ảnh
    if combined_map.dim() <= 3 or (combined_map.dim() == 4 and combined_map.shape[0] == 1):
        k_pixels = max(1, int(top_k_ratio * combined_map.numel()))
        img_score = torch.topk(combined_map.flatten(), k=k_pixels).values.mean().item()
        return combined_map, img_score
    else:
        b = combined_map.shape[0]
        flat = combined_map.view(b, -1)
        k_pixels = max(1, int(top_k_ratio * flat.shape[1]))
        img_scores = torch.topk(flat, k=k_pixels, dim=1).values.mean(dim=1)
        return combined_map, img_scores


def apply_global_thresholding(
    anomaly_map: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Apply Global Baseline Thresholding.
    
    Formula:
        \\tilde{M}_l = ReLU( M_l - \\tau_l^{global} )
        
    Args:
        anomaly_map: Discrepancy map [B, 1, H, W]
        threshold: Global baseline scalar threshold \\tau_l^{global}
    """
    return F.relu(anomaly_map - threshold)


def apply_quantile_soft_thresholding(
    anomaly_map: torch.Tensor,
    quantile_ratio: float = 0.60,
) -> torch.Tensor:
    """Apply Per-sample Soft-Thresholding background noise filter.
    
    Formula:
        \\tilde{M}_l = ReLU( M_l - Quantile_{0.60}(M_l) )
        
    Strips out 60% background noise in each layer, preserving the top 40% highest
    discrepancy responses.
    
    Args:
        anomaly_map: Tensor of shape [B, 1, H, W]
        quantile_ratio: Quantile cutoff (default 0.60, keeping top 40%)
    """
    if quantile_ratio <= 0.0:
        return F.relu(anomaly_map)

    b = anomaly_map.shape[0]
    filtered_maps = []

    for i in range(b):
        m_item = anomaly_map[i, 0]  # [H, W]
        # Calculate quantile across spatial dimensions
        q_val = torch.quantile(m_item.float(), q=quantile_ratio)
        # Soft-threshold: ReLU(M - Quantile)
        m_thresh = F.relu(m_item - q_val)
        filtered_maps.append(m_thresh.unsqueeze(0).unsqueeze(0).to(anomaly_map.dtype))

    return torch.cat(filtered_maps, dim=0)


@torch.no_grad()
def compute_global_baseline_thresholds(
    model: nn.Module,
    val_loader: Any,
    layers: Optional[List[str]] = None,
    quantile: float = 0.95,
    device: Optional[Union[str, torch.device]] = None,
    max_samples: Optional[int] = 200,
    max_pixels_per_layer: int = 1_000_000,
) -> Dict[str, float]:
    """Compute fixed global baseline thresholds \\tau_l^{global} on normal validation set.
    
    Formula:
        \\tau_l^{global} = Quantile_{0.95}\\Big( \\bigcup_{X \\in \\mathcal{D}_{val\\_norm}} M_l(X) \\Big)
        
    For normal images X_{norm}, almost all pixels have M_l(X_{norm}) < \\tau_l^{global},
    yielding:
        ReLU(M_l - \\tau_l^{global}) = 0  ==>  A(X_{norm}) \\approx 0.0
        
    Args:
        model: Trained PatchCL_AE model
        val_loader: DataLoader containing validation/training normal samples
        layers: List of feature layers (default: E_1 .. E_4)
        quantile: Quantile cutoff across all pooled normal pixels (default: 0.95)
        device: Device to run inference on
        max_samples: Maximum normal samples to pool from val_loader (default: 200 for fast & robust estimate)
        max_pixels_per_layer: Maximum pixels to store per layer to prevent memory spike
        
    Returns:
        Dict mapping each layer_name to float threshold: {layer: \\tau_l^{global}}
    """
    model.eval()
    if layers is None:
        layers = getattr(model, "feature_layers", ["E_1", "E_2", "E_3", "E_4"])

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    layer_pixel_pools: Dict[str, List[torch.Tensor]] = {l: [] for l in layers}
    layer_pixel_counts: Dict[str, int] = {l: 0 for l in layers}

    sample_count = 0

    for batch in val_loader:
        # Determine images and labels
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch.get("label", None)
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            labels = None

        # Filter only normal samples (label == 0) if labels exist
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                normal_mask = (labels == 0)
            elif isinstance(labels, (int, float)):
                normal_mask = torch.tensor([labels == 0], dtype=torch.bool)
            else:
                normal_mask = torch.tensor([lbl == 0 for lbl in labels], dtype=torch.bool)
            
            if not normal_mask.any():
                continue
            images = images[normal_mask]

        images = images.to(device, non_blocking=True)
        if images.dim() == 5 and images.shape[1] == 1:
            images = images.squeeze(1)
        b = images.shape[0]

        # Forward AE to get reconstructed output and features
        x_recon = model(images)
        if hasattr(model, "encode"):
            _, feats_recon = model.encode(x_recon)
            _, feats_real = model.encode(images)
        else:
            _, feats_recon = model.encoder(x_recon, return_features=True)
            _, feats_real = model.encoder(images, return_features=True)

        for layer_name in layers:
            if layer_pixel_counts[layer_name] >= max_pixels_per_layer:
                continue

            feat_real = feats_real[layer_name]
            feat_recon = feats_recon[layer_name]
            proj_head = model.projection.heads[layer_name]

            # Cosine discrepancy map M_l: [B, 1, H_l, W_l]
            m_l = compute_layer_similarity_map(feat_real, feat_recon, proj_head)
            
            # Flatten to 1D and move to CPU
            flat_pixels = m_l.detach().cpu().flatten()
            
            # Check remaining budget
            rem = max_pixels_per_layer - layer_pixel_counts[layer_name]
            if flat_pixels.numel() > rem:
                flat_pixels = flat_pixels[:rem]

            layer_pixel_pools[layer_name].append(flat_pixels)
            layer_pixel_counts[layer_name] += flat_pixels.numel()

        sample_count += b
        if max_samples is not None and sample_count >= max_samples:
            break

    # Calculate quantile for each layer
    global_thresholds: Dict[str, float] = {}
    for layer_name in layers:
        pools = layer_pixel_pools[layer_name]
        if len(pools) == 0:
            global_thresholds[layer_name] = 0.0
            continue

        pooled_tensor = torch.cat(pools, dim=0)
        q_val = float(torch.quantile(pooled_tensor.float(), q=quantile).item())
        global_thresholds[layer_name] = q_val

    return global_thresholds


def save_global_thresholds(thresholds: Dict[str, float], file_path: str) -> None:
    """Save global baseline thresholds to a JSON or PT file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if file_path.endswith(".pt") or file_path.endswith(".pth"):
        torch.save(thresholds, file_path)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)


def load_global_thresholds(file_path: str) -> Dict[str, float]:
    """Load global baseline thresholds from a JSON or PT file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Thresholds file not found: {file_path}")
    if file_path.endswith(".pt") or file_path.endswith(".pth"):
        data = torch.load(file_path, map_location="cpu")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def compute_patchcl_anomaly_map(
    model: nn.Module,
    x: torch.Tensor,
    layers: Optional[List[str]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    layer_weights: Optional[Union[Dict[str, float], List[float]]] = None,
    top_k_ratio: float = 0.10,
    quantile_ratio: Optional[float] = None,
    global_thresholds: Optional[Dict[str, float]] = None,
    use_thresholding: bool = False,
    use_projection_head: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute overall patch-wise anomaly score map directly WITHOUT soft-thresholding.
    
    Formula:
        M_l = 1 - cos(P_l(E_l(X)), P_l(E_l(X_hat)))
        M_{combined} = sum_{l=1}^L w_l * resize(M_l)
        
    Args:
        model: DecoupledPatchCLAE / PatchCL_AE instance
        x: Raw input test images [B, 3, H, W]
        layers: List of encoder layers to compute discrepancy on (default: E_1 .. E_4 or E_1 .. E_5)
        target_size: Output spatial resolution (default: (H, W) of x)
        layer_weights: Custom layer weighting dictionary or list
        top_k_ratio: Top percentage of pixels for scoring (default: 0.10)
        quantile_ratio: (Deprecated) Kept for fallback compatibility if use_thresholding=True
        global_thresholds: (Optional) Kept for fallback compatibility if use_thresholding=True
        use_thresholding: If True, applies thresholding. Default: False.
        use_projection_head: If True, uses P_l to project features before computing cosine distance (Paper Eq. 5, 6).
        
    Returns:
        combined_map: Combined anomaly heatmap M [B, 1, H, W]
        x_recon: Reconstructed image [B, 3, H, W]
        layer_maps: Dictionary of individual raw layer maps {layer_name: M_l}
    """
    model.eval()
    if target_size is None:
        target_size = (x.shape[2], x.shape[3])

    if layers is None:
        layers = getattr(model, "feature_layers", ["E_1", "E_2", "E_3", "E_4"])

    if isinstance(layer_weights, dict):
        weights_list = [layer_weights.get(l, 1.0 / len(layers)) for l in layers]
    elif isinstance(layer_weights, (list, tuple)):
        weights_list = list(layer_weights)
    elif len(layers) == 5:
        # Chuẩn bài báo Lu et al. 2024: Cộng dồn đều cả 5 tầng (Eq. 7)
        weights_list = [1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        weights_list = [0.05, 0.15, 0.35, 0.45]

    with torch.no_grad():
        # 1. Forward AE to get reconstructed output and features
        x_recon = model(x)
        if hasattr(model, "encode"):
            _, feats_recon = model.encode(x_recon)
            _, feats_real = model.encode(x)
        else:
            _, feats_recon = model.encoder(x_recon, return_features=True)
            _, feats_real = model.encoder(x, return_features=True)

        raw_layer_maps: Dict[str, torch.Tensor] = {}
        resized_maps_list: List[torch.Tensor] = []

        # 2. Compute similarity for each layer
        for layer_name in layers:
            feat_real = feats_real[layer_name]
            feat_recon = feats_recon[layer_name]
            proj_head = model.projection.heads[layer_name] if hasattr(model, "projection") and layer_name in model.projection.heads else None

            # Cosine discrepancy map M_l = 1 - cos_sim
            m_l = compute_layer_similarity_map(feat_real, feat_recon, proj_head, use_projection_head=use_projection_head)
            raw_layer_maps[layer_name] = m_l

            # Áp dụng thresholding chỉ khi người dùng chỉ định use_thresholding=True
            if use_thresholding:
                if global_thresholds is not None and layer_name in global_thresholds:
                    m_l = apply_global_thresholding(m_l, threshold=float(global_thresholds[layer_name]))
                elif quantile_ratio is not None and quantile_ratio > 0:
                    m_l = apply_quantile_soft_thresholding(m_l, quantile_ratio=quantile_ratio)

            # Bilinear interpolation to original image size
            m_l_resized = F.interpolate(
                m_l, size=target_size, mode="bilinear", align_corners=True
            )
            resized_maps_list.append(m_l_resized)

        # 3. Direct summation across 4 layers with deep weights
        combined_map, _ = compute_anomaly_score_simple(
            layer_maps=resized_maps_list,
            weights=weights_list,
            top_k_ratio=top_k_ratio,
            target_size=target_size,
        )

    return combined_map, x_recon, raw_layer_maps


def anomaly_map_to_score(
    anomaly_map: torch.Tensor,
    reduction: str = "top_percent",
    top_k_ratio: float = 0.10,
    top_k: int = 100,
    border_margin: int = 10,
) -> torch.Tensor:
    """Reduce 2D anomaly map M to image-level anomaly score.
    
    Default is Top 10% pixel mean ('top_percent' with top_k_ratio=0.10).
    Excludes the outer border margin (default: 10px) to prevent zero-padding convolution
    boundary artifacts from corrupting the anomaly score.
    
    Args:
        anomaly_map: [B, 1, H, W]
        reduction: "top_percent", "top_k", "mean", "max"
        top_k_ratio: Ratio of highest pixels to average if reduction == "top_percent" (default 0.10)
        top_k: Number of highest pixels to average if reduction == "top_k"
        border_margin: Outer border margin in pixels to exclude (default: 10)
    """
    if border_margin > 0 and anomaly_map.shape[-2] > 2 * border_margin and anomaly_map.shape[-1] > 2 * border_margin:
        am_crop = anomaly_map[..., border_margin:-border_margin, border_margin:-border_margin].contiguous()
    else:
        am_crop = anomaly_map

    flat = am_crop.reshape(am_crop.shape[0], -1)
    if reduction in ("top_percent", "top_10_percent"):
        k = max(1, int(top_k_ratio * flat.size(1)))
        top_vals, _ = torch.topk(flat, k=k, dim=1)
        return top_vals.mean(dim=1)
    elif reduction == "mean":
        return flat.mean(dim=1)
    elif reduction == "max":
        return flat.max(dim=1).values
    elif reduction == "top_k":
        k = min(top_k, flat.size(1))
        top_vals, _ = torch.topk(flat, k=k, dim=1)
        return top_vals.mean(dim=1)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")
