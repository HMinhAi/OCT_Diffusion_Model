"""Evaluation and Heatmap Visualization script for Decoupled Dual-Stream PatchCL-AE.

Generates:
    - Anomaly classification metrics (AUC, F1, ACC, SEN, SPE)
    - Visualization heatmaps with soft-thresholding noise suppression
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

# Ensure headless Agg backend before importing pyplot to prevent GUI hangs on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure parent directory is in sys.path when running as a standalone script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .anomaly_score import (
        anomaly_map_to_score,
        compute_global_baseline_thresholds,
        compute_patchcl_anomaly_map,
        load_global_thresholds,
        save_global_thresholds,
    )
    from .dataset import create_patchcl_dataloaders
    from .metrics import calculate_anomaly_metrics
    from .models import create_patchcl_ae_models
except (ImportError, ValueError):
    from PatchCL_AE_MHSA.anomaly_score import (
        anomaly_map_to_score,
        compute_global_baseline_thresholds,
        compute_patchcl_anomaly_map,
        load_global_thresholds,
        save_global_thresholds,
    )
    from PatchCL_AE_MHSA.dataset import create_patchcl_dataloaders
    from PatchCL_AE_MHSA.metrics import calculate_anomaly_metrics
    from PatchCL_AE_MHSA.models import create_patchcl_ae_models


def save_heatmap_visualization(
    raw_img: np.ndarray,
    recon_img: np.ndarray,
    anomaly_map: np.ndarray,
    save_path: str,
    label: int,
    score: float,
) -> None:
    """Save raw image, reconstructed image, anomaly heatmap, and overlay."""
    raw_img = np.asarray(raw_img, dtype=np.float32)
    recon_img = np.asarray(recon_img, dtype=np.float32)
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    # Denormalize from [-1, 1] to [0, 1]
    raw_disp = np.clip((raw_img.transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
    recon_disp = np.clip((recon_img.transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)

    # Normalize heatmap to [0, 1] with numerical noise safeguard
    am = anomaly_map[0]
    am_min, am_max = float(am.min()), float(am.max())
    diff = am_max - am_min
    if diff > 1e-4:
        am_norm = (am - am_min) / diff
    else:
        am_norm = np.zeros_like(am)
    heatmap_colored = plt.get_cmap("jet")(am_norm)[:, :, :3].astype(np.float32)

    # Overlay
    overlay = np.clip(0.6 * raw_disp + 0.4 * heatmap_colored, 0.0, 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(raw_disp)
    axes[0].set_title(f"Raw Input ({'Anomaly' if label == 1 else 'Normal'})")
    axes[0].axis("off")

    axes[1].imshow(recon_disp)
    axes[1].set_title("Reconstructed Output")
    axes[1].axis("off")

    im = axes[2].imshow(am_norm, cmap="jet")
    axes[2].set_title(f"Anomaly Map ({score:.4f})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay Heatmap")
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    plt.close("all")


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    data_root: str,
    img_size: int = 224,
    resize_size: int = 256,
    batch_size: int = 16,
    num_heads: int = 4,
    gamma: float = 2.0,
    quantile_ratio: float = 0.60,
    device: Optional[str] = None,
    save_visualizations: bool = True,
    viz_dir: str = "visualizations",
    max_viz_samples: int = 20,
    num_workers: int = 0,
    reduction: str = "top_percent",
    top_k_ratio: float = 0.10,
    top_k: int = 100,
    use_global_thresholding: bool = False,
    global_threshold_quantile: float = 0.95,
    global_thresholds: Optional[Dict[str, float]] = None,
    thresholds_cache_path: Optional[str] = None,
    **kwargs,  # Gracefully ignore deprecated arguments
) -> Dict[str, float]:
    """Load model from checkpoint and evaluate on test set."""
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"\n[Decoupled PatchCL-AE Evaluation] Loading data from '{data_root}'...", flush=True)
    loaders = create_patchcl_dataloaders(
        root_path=data_root,
        img_size=img_size,
        resize_size=resize_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = loaders["test"]

    print(f"Loading checkpoint '{checkpoint_path}' on device '{dev}'...", flush=True)
    generator, _ = create_patchcl_ae_models(
        in_channels=3,
        num_heads=num_heads,
        gamma=gamma,
        mhsa_mode=kwargs.get("mhsa_mode", "bottleneck"),
    )

    try:
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=dev)

    if isinstance(ckpt, dict) and "generator_state" in ckpt:
        generator.load_state_dict(ckpt["generator_state"], strict=False)
        epoch = ckpt.get("epoch", "unknown")
        print(f"Checkpoint successfully loaded (Trained Epoch: {epoch})", flush=True)
    elif isinstance(ckpt, dict):
        generator.load_state_dict(ckpt, strict=False)
        print("Model state dict successfully loaded.", flush=True)
    else:
        generator = ckpt

    generator.to(dev)
    generator.eval()

    # Resolve Global Baseline Thresholds
    if use_global_thresholding:
        if global_thresholds is None:
            # Check if saved in checkpoint
            if isinstance(ckpt, dict) and "global_thresholds" in ckpt and ckpt["global_thresholds"] is not None:
                global_thresholds = ckpt["global_thresholds"]
                print(f"[Global Thresholds] Loaded precomputed thresholds from checkpoint: { {k: round(v, 4) for k, v in global_thresholds.items()} }", flush=True)
            elif thresholds_cache_path and os.path.exists(thresholds_cache_path):
                global_thresholds = load_global_thresholds(thresholds_cache_path)
                print(f"[Global Thresholds] Loaded thresholds from '{thresholds_cache_path}': { {k: round(v, 4) for k, v in global_thresholds.items()} }", flush=True)
            else:
                print(f"[Global Baseline Thresholding] Computing tau_l^global on normal validation set (q={global_threshold_quantile})...", flush=True)
                global_thresholds = compute_global_baseline_thresholds(
                    model=generator,
                    val_loader=loaders["train"],
                    quantile=global_threshold_quantile,
                    device=dev,
                    max_samples=200,
                )
                print(f"[Global Baseline Thresholding] Computed thresholds: { {k: round(v, 4) for k, v in global_thresholds.items()} }", flush=True)
                if thresholds_cache_path:
                    save_global_thresholds(global_thresholds, thresholds_cache_path)
                    print(f"[Global Baseline Thresholding] Saved to: {thresholds_cache_path}", flush=True)

    all_labels = []
    all_scores = []
    viz_count = 0

    num_test = len(test_loader.dataset)
    print(f"Evaluating {num_test} images across {len(test_loader)} batches...", flush=True)

    pbar = tqdm(
        test_loader,
        desc="Evaluating",
        unit="batch",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for batch in pbar:
        images = batch["image"].to(dev, non_blocking=True)
        labels = batch["label"]

        anomaly_maps, recons, _ = compute_patchcl_anomaly_map(
            model=generator,
            x=images,
            top_k_ratio=top_k_ratio,
            quantile_ratio=quantile_ratio,
            global_thresholds=global_thresholds if use_global_thresholding else None,
            use_thresholding=use_global_thresholding,
        )
        scores = anomaly_map_to_score(
            anomaly_maps,
            reduction=reduction,
            top_k_ratio=top_k_ratio,
            top_k=top_k,
        )

        all_labels.extend(labels.cpu().numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())

        if save_visualizations and viz_count < max_viz_samples:
            for i in range(images.shape[0]):
                if viz_count >= max_viz_samples:
                    break
                raw_np = images[i].cpu().numpy()
                recon_np = recons[i].cpu().numpy()
                am_np = anomaly_maps[i].cpu().numpy()
                lbl = int(labels[i].item())
                sc = float(scores[i].item())

                save_p = os.path.join(viz_dir, f"sample_{viz_count:04d}_lbl_{lbl}.png")
                save_heatmap_visualization(raw_np, recon_np, am_np, save_p, lbl, sc)
                viz_count += 1

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    if (labels == 0).any():
        print(f"Normal scores  : min={scores[labels==0].min():.3f}, mean={scores[labels==0].mean():.3f}, max={scores[labels==0].max():.3f}", flush=True)
    if (labels == 1).any():
        print(f"Abnormal scores: min={scores[labels==1].min():.3f}, mean={scores[labels==1].mean():.3f}, max={scores[labels==1].max():.3f}", flush=True)

    metrics = calculate_anomaly_metrics(all_labels, all_scores)
    print("\n" + "=" * 50, flush=True)
    print("PATCHCL-AE EVALUATION RESULTS:", flush=True)
    print(f"  - AUC:         {metrics['auc']:.2f}%", flush=True)
    print(f"  - F1-Score:    {metrics['f1']:.2f}%", flush=True)
    print(f"  - Accuracy:    {metrics['acc']:.2f}%", flush=True)
    print(f"  - Sensitivity: {metrics['sen']:.2f}%", flush=True)
    print(f"  - Specificity: {metrics['spe']:.2f}%", flush=True)
    print(f"  - Threshold:   {metrics['best_threshold']:.6f}", flush=True)
    if save_visualizations:
        print(f"  - Saved {viz_count} heatmaps to '{viz_dir}'", flush=True)
    print("=" * 50 + "\n", flush=True)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Decoupled Dual-Stream PatchCL-AE model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_root", type=str, default="datasets/OCT2017", help="Root data folder (default: datasets/OCT2017)")
    parser.add_argument("--img_size", type=int, default=224, help="Image crop resolution (default: 224)")
    parser.add_argument("--resize_size", type=int, default=256, help="Image resize resolution (default: 256)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--mhsa_mode", type=str, default="bottleneck", choices=["bottleneck", "parallel", "both"], help="MHSA mode")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Attenuation exponent gamma (default: 2.0)")
    parser.add_argument("--quantile_ratio", type=float, default=0.60, help="Quantile cutoff for soft-thresholding (default: 0.60)")
    parser.add_argument("--reduction", type=str, default="top_percent", choices=["max", "top_percent", "top_k", "mean"], help="Reduction method (default: top_percent)")
    parser.add_argument("--top_k_ratio", type=float, default=0.10, help="Top percentage ratio for scoring (default: 0.10)")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top pixels for top_k (default: 100)")
    parser.add_argument("--use_global_thresholding", action="store_true", default=False, help="Enable Global Baseline Thresholding")
    parser.add_argument("--global_quantile", type=float, default=0.95, help="Quantile cutoff for global baseline thresholding (default: 0.95)")
    parser.add_argument("--thresholds_path", type=str, default=None, help="Path to cache/load global thresholds (.json or .pt)")
    parser.add_argument("--viz_dir", type=str, default="visualizations/patchcl_ae_mhsa", help="Visualization directory")
    parser.add_argument("--max_viz", type=int, default=20, help="Max visualization samples")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (0 recommended on Windows)")

    # Deprecated args
    parser.add_argument("--num_mhsa_layers", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--num_prototypes", type=int, default=800, help=argparse.SUPPRESS)
    parser.add_argument("--top_p", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--tau_m", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--d_p", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--sigma_sem", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--gamma_pos", type=float, default=0.5, help=argparse.SUPPRESS)

    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        img_size=args.img_size,
        resize_size=args.resize_size,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        gamma=args.gamma,
        quantile_ratio=args.quantile_ratio,
        reduction=args.reduction,
        top_k_ratio=args.top_k_ratio,
        top_k=args.top_k,
        mhsa_mode=args.mhsa_mode,
        use_global_thresholding=args.use_global_thresholding,
        global_threshold_quantile=args.global_quantile,
        thresholds_cache_path=args.thresholds_path,
        viz_dir=args.viz_dir,
        max_viz_samples=args.max_viz,
        num_workers=args.num_workers,
    )
