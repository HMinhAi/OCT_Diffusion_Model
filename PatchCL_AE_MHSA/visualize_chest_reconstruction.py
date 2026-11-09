"""Visualization and Reconstruction Quality Assessment for Chest X-Ray Anomaly Detection.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from PatchCL_AE_MHSA.dataset import get_patchcl_transforms
from PatchCL_AE_MHSA.anomaly_score import compute_patchcl_anomaly_map, anomaly_map_to_score


def evaluate_and_visualize_chest_samples(
    model: torch.nn.Module,
    data_root: str = "datasets/chest_xray",
    output_path: str = "visualizations/patchcl_ae_mhsa/chest_reconstruction_eval.png",
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluates and visualizes reconstructions on NORMAL, BACTERIA, and VIRUS Chest X-Ray samples."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    transform = get_patchcl_transforms(img_size=224, resize_size=256, is_train=False)

    # Pick representative samples
    norm_files = sorted(glob.glob(os.path.join(data_root, "test", "NORMAL", "*.jpeg")))[:2]
    bac_files = sorted(glob.glob(os.path.join(data_root, "test", "PNEUMONIA", "BACTERIA*.jpeg")))[:2]
    vir_files = sorted(glob.glob(os.path.join(data_root, "test", "PNEUMONIA", "VIRUS*.jpeg")))[:2]

    samples_to_test = (
        [(f, "NORMAL", False) for f in norm_files]
        + [(f, "BACTERIAL PNEUMONIA", True) for f in bac_files]
        + [(f, "VIRAL PNEUMONIA", True) for f in vir_files]
    )

    rows_data = []
    scores_by_type: Dict[str, List[float]] = {"NORMAL": [], "BACTERIAL PNEUMONIA": [], "VIRAL PNEUMONIA": []}
    l1_by_type: Dict[str, List[float]] = {"NORMAL": [], "BACTERIAL PNEUMONIA": [], "VIRAL PNEUMONIA": []}

    for fpath, label_name, is_anomaly in samples_to_test:
        pil_img = Image.open(fpath).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            anomaly_map, x_rec, _ = compute_patchcl_anomaly_map(
                model=model,
                x=img_tensor,
                layers=["E_1", "E_2", "E_3", "E_4"],
                top_k_ratio=0.10,
                use_thresholding=False,
                use_projection_head=False,
            )
            score = anomaly_map_to_score(
                anomaly_map,
                reduction="top_percent",
                top_k_ratio=0.10,
            ).item()

            l1_err = F.l1_loss(x_rec, img_tensor).item()
            scores_by_type[label_name].append(score)
            l1_by_type[label_name].append(l1_err)

            raw_disp = np.clip((img_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
            rec_disp = np.clip((x_rec[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
            diff_disp = np.clip(np.abs(raw_disp - rec_disp), 0.0, 1.0)

            # Pixel Reconstruction Residual Heatmap |X - X_hat|
            diff_2d = np.mean(diff_disp, axis=-1)  # [H, W]
            m = 10
            mask = np.zeros_like(diff_2d, dtype=bool)
            mask[m:-m, m:-m] = True
            diff_clean = diff_2d.copy()
            diff_clean[~mask] = 0.0

            # Gaussian smoothing (sigma=3.5) to form contiguous anatomical heatmap
            diff_smooth = gaussian_filter(diff_clean, sigma=3.5)
            peak_res = float(diff_smooth.max())

            # Normalize with standard medical dynamic range [0.0, 0.22]
            res_norm = np.clip(diff_smooth / 0.22, 0.0, 1.0)

            # Adaptive alpha overlay: low-noise normal lungs (< 0.22) remain clean/transparent
            am_color = plt.get_cmap("jet")(res_norm)[:, :, :3]
            alpha = np.clip((res_norm - 0.22) / 0.78, 0.0, 0.70)[:, :, None]
            overlay = np.clip((1.0 - alpha) * raw_disp + alpha * am_color, 0.0, 1.0)

            rows_data.append({
                "type": label_name,
                "is_anomaly": is_anomaly,
                "filename": os.path.basename(fpath),
                "raw": raw_disp,
                "rec": rec_disp,
                "diff": diff_disp,
                "am_norm": res_norm,
                "diff_smooth": diff_smooth,
                "peak_res": peak_res,
                "overlay": overlay,
                "l1": l1_err,
                "score": score,
            })

    # Summary dict
    results_summary = {
        k: {"avg_l1": float(np.mean(l1_by_type[k])), "avg_score": float(np.mean(scores_by_type[k]))}
        for k in scores_by_type if scores_by_type[k]
    }

    # Plot
    n_rows = len(rows_data)
    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 3.8 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        for idx, item in enumerate(rows_data):
            is_anomaly = item["is_anomaly"]
            col_tag = "red" if is_anomaly else "green"

            # Col 1: Raw
            axes[idx, 0].imshow(item["raw"])
            axes[idx, 0].set_title(f"Input: {item['type']}\n{item['filename']}", fontsize=11, fontweight="bold")
            axes[idx, 0].axis("off")

            # Col 2: Reconstructed
            axes[idx, 1].imshow(item["rec"])
            axes[idx, 1].set_title(f"MHSA-AE Reconstructed\n(L1: {item['l1']:.4f})", fontsize=11)
            axes[idx, 1].axis("off")

            # Col 3: Pixel Difference
            axes[idx, 2].imshow(item["diff"], cmap="gray", vmin=0.0, vmax=0.35)
            axes[idx, 2].set_title("Abs Difference |X - X_hat|", fontsize=11)
            axes[idx, 2].axis("off")

            # Col 4: Residual Heatmap
            im = axes[idx, 3].imshow(item["diff_smooth"], cmap="jet", vmin=0.0, vmax=0.25)
            axes[idx, 3].set_title(f"Residual Heatmap |X - X_hat|\nPeak: {item['peak_res']:.4f}", fontsize=11)
            axes[idx, 3].axis("off")
            plt.colorbar(im, ax=axes[idx, 3], fraction=0.046, pad=0.04)

            # Col 5: Overlay
            axes[idx, 4].imshow(item["overlay"])
            axes[idx, 4].set_title(
                f"{'[PNEUMONIA DETECTED]' if is_anomaly else '[NORMAL LUNGS]'}",
                fontsize=11,
                color=col_tag,
                fontweight="bold",
            )
            axes[idx, 4].axis("off")

        plt.suptitle("Chest X-Ray Anomaly Detection & Reconstruction Quality (PatchCL-AE + MHSA)", fontsize=16, fontweight="bold", y=1.002)
        plt.tight_layout()
        
        abs_output_path = os.path.normpath(os.path.abspath(output_path))
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        try:
            plt.savefig(abs_output_path, bbox_inches="tight", dpi=160)
            print(f"[Visualization Saved] -> {abs_output_path}", flush=True)
        except Exception as e:
            print(f"[Visualization Warning] Could not save {abs_output_path}: {e}", flush=True)
        finally:
            plt.close(fig)
            plt.close("all")

    return results_summary


if __name__ == "__main__":
    from PatchCL_AE_MHSA.models import create_patchcl_ae_models
    gen, _ = create_patchcl_ae_models(in_channels=3)
    res = evaluate_and_visualize_chest_samples(gen, output_path="visualizations/test_chest_init.png")
    print("Initial sanity chest results:", res)
