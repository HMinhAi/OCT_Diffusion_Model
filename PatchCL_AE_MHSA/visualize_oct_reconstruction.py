"""Visualization and Reconstruction Quality Assessment for Retinal OCT Anomaly Detection.
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
import torch
import torch.nn.functional as F

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from PatchCL_AE_MHSA.dataset import get_patchcl_transforms
from PatchCL_AE_MHSA.anomaly_score import compute_patchcl_anomaly_map, anomaly_map_to_score


def evaluate_and_visualize_oct_samples(
    model: torch.nn.Module,
    data_root: str = "datasets/OCT2017",
    output_path: str = "visualizations/oct_reconstruction_eval.png",
    device: Optional[torch.device] = None,
    num_samples_per_class: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Evaluates and visualizes reconstructions across NORMAL, CNV, DME, and DRUSEN classes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    classes = ["NORMAL", "CNV", "DME", "DRUSEN"]
    transform = get_patchcl_transforms(img_size=224, resize_size=256, is_train=False)

    results_by_class: Dict[str, Dict[str, float]] = {}
    rows_data = []

    for cls in classes:
        pattern = os.path.join(data_root, "test", cls, "*.jpeg")
        files = sorted(glob.glob(pattern))
        if not files:
            # Fallback to train if test empty
            pattern = os.path.join(data_root, "train", cls, "*.jpeg")
            files = sorted(glob.glob(pattern))
        if not files:
            continue

        selected_files = files[:num_samples_per_class]

        l1_errors = []
        scores_list = []

        for fpath in selected_files:
            pil_img = Image.open(fpath).convert("RGB")
            img_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

            with torch.no_grad():
                # Forward model to get reconstruction and anomaly map
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
                l1_errors.append(l1_err)
                scores_list.append(score)

                # Prepare numpy arrays for plotting
                # Tensor in [-1, 1] -> [0, 1]
                raw_disp = np.clip((img_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
                rec_disp = np.clip((x_rec[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
                diff_disp = np.clip(np.abs(raw_disp - rec_disp), 0.0, 1.0)

                am = anomaly_map[0, 0].cpu().numpy()
                am_norm = (am - am.min()) / (am.max() - am.min() + 1e-8)
                am_color = plt.get_cmap("jet")(am_norm)[:, :, :3]
                overlay = np.clip(0.6 * raw_disp + 0.4 * am_color, 0.0, 1.0)

                rows_data.append({
                    "class": cls,
                    "filename": os.path.basename(fpath),
                    "raw": raw_disp,
                    "rec": rec_disp,
                    "diff": diff_disp,
                    "am_norm": am_norm,
                    "overlay": overlay,
                    "l1": l1_err,
                    "score": score,
                })

        results_by_class[cls] = {
            "avg_l1": float(np.mean(l1_errors)),
            "avg_score": float(np.mean(scores_list)),
        }

    # Plot the figure
    n_rows = len(rows_data)
    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        for idx, item in enumerate(rows_data):
            cls_name = item["class"]
            is_anomaly = (cls_name != "NORMAL")

            # Col 1: Raw
            axes[idx, 0].imshow(item["raw"])
            axes[idx, 0].set_title(f"Input: {cls_name} ({'Anomaly' if is_anomaly else 'Normal'})\n{item['filename']}", fontsize=11, fontweight="bold")
            axes[idx, 0].axis("off")

            # Col 2: Reconstructed
            axes[idx, 1].imshow(item["rec"])
            axes[idx, 1].set_title(f"MHSA-AE Reconstructed\n(L1: {item['l1']:.4f})", fontsize=11)
            axes[idx, 1].axis("off")

            # Col 3: Pixel Difference
            axes[idx, 2].imshow(item["diff"])
            axes[idx, 2].set_title("Abs Difference |X - X̂|", fontsize=11)
            axes[idx, 2].axis("off")

            # Col 4: Anomaly Discrepancy Map
            im = axes[idx, 3].imshow(item["am_norm"], cmap="jet")
            axes[idx, 3].set_title(f"Anomaly Heatmap\nScore: {item['score']:.4f}", fontsize=11)
            axes[idx, 3].axis("off")
            plt.colorbar(im, ax=axes[idx, 3], fraction=0.046, pad=0.04)

            # Col 5: Overlay
            axes[idx, 4].imshow(item["overlay"])
            axes[idx, 4].set_title(f"Overlay Localization\n{'[LESTION DETECTED]' if is_anomaly else '[NORMAL RETINA]'}", fontsize=11, color="red" if is_anomaly else "green", fontweight="bold")
            axes[idx, 4].axis("off")

        plt.suptitle("Retinal OCT Anomaly Detection & Reconstruction Quality (PatchCL-AE + MHSA)", fontsize=16, fontweight="bold", y=1.002)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        plt.close("all")
        print(f"[Visualization Saved] -> {output_path}")

    return results_by_class


if __name__ == "__main__":
    from PatchCL_AE_MHSA.models import create_patchcl_ae_models
    gen, _ = create_patchcl_ae_models(in_channels=3)
    res = evaluate_and_visualize_oct_samples(gen, output_path="visualizations/test_oct_init.png")
    print("Initial sanity results by class:", res)
