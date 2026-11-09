"""Training CLI script for MHSA-Enhanced PatchCL-AE.

Usage:
    python -m PatchCL_AE_MHSA.train --data_root datasets/OCT2017 --epochs 15 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Ensure parent directory is in sys.path when running as a standalone script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .dataset import create_patchcl_dataloaders
    from .models import create_patchcl_ae_models
    from .trainer import PatchCLTrainer
except (ImportError, ValueError):
    from PatchCL_AE_MHSA.dataset import create_patchcl_dataloaders
    from PatchCL_AE_MHSA.models import create_patchcl_ae_models
    from PatchCL_AE_MHSA.trainer import PatchCLTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MHSA-Enhanced PatchCL-AE for Retinal OCT Anomaly Detection."
    )

    # 1. Dataset & Resolution
    data_group = parser.add_argument_group("Dataset & Resolution")
    data_group.add_argument(
        "--data_root",
        type=str,
        default="datasets/OCT2017",
        help="Path to dataset root (default: datasets/OCT2017)",
    )
    data_group.add_argument("--img_size", type=int, default=224, help="Network crop resolution (default: 224)")
    data_group.add_argument("--resize_size", type=int, default=256, help="Initial resize resolution (default: 256)")
    data_group.add_argument("--num_workers", type=int, default=2, help="DataLoader workers count (default: 2)")
    data_group.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples (None = all, e.g. 2000 for fast testing)")
    data_group.add_argument("--max_test_samples", type=int, default=None, help="Limit test samples (None = all, e.g. 400)")

    # 2. Optimization & Hardware
    opt_group = parser.add_argument_group("Optimization & Hardware")
    opt_group.add_argument("--epochs", type=int, default=15, help="Max training epochs (default: 15)")
    opt_group.add_argument("--batch_size", type=int, default=16, help="Minibatch size (default: 16)")
    opt_group.add_argument("--lr", type=float, default=0.0002, help="Initial Adam learning rate for Generator (default: 0.0002)")
    opt_group.add_argument("--lr_d", type=float, default=0.000001, help="Learning rate for Discriminator (default: 0.000001)")
    opt_group.add_argument("--beta1", type=float, default=0.9, help="Adam beta1 (default: 0.9)")
    opt_group.add_argument("--beta2", type=float, default=0.999, help="Adam beta2 (default: 0.999)")
    opt_group.add_argument("--eps", type=float, default=1e-4, help="Adam epsilon (default: 0.0001)")
    opt_group.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    opt_group.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')")

    # 3. Model Architecture & MHSA
    model_group = parser.add_argument_group("Architecture & MHSA")
    model_group.add_argument(
        "--mhsa_mode",
        type=str,
        default="bottleneck",
        choices=["bottleneck", "parallel", "both"],
        help="MHSA integration mode: 'bottleneck' (in reconstruction path), 'parallel', or 'both'",
    )
    model_group.add_argument("--num_heads", type=int, default=4, help="Attention heads in MHSA (default: 4)")
    model_group.add_argument("--gamma", type=float, default=2.0, help="Exponent gamma for attention attenuation w(i, j) = (1 - S_{ij})^gamma (default: 2.0)")
    model_group.add_argument("--noise_std", type=float, default=0.05, help="Gaussian noise std added to inputs (default: 0.05)")
    model_group.add_argument("--feature_layers", type=str, nargs="+", default=["E_1", "E_2", "E_3", "E_4"], help="Encoder layers for contrastive loss")
    model_group.add_argument("--num_conv_bottlenecks", type=int, default=6, help="Number of conv blocks in E_5 lossy bottleneck (default: 6)")

    # 4. Loss Objectives
    loss_group = parser.add_argument_group("Loss Functions & Weights")
    loss_group.add_argument("--lambda_adv", type=float, default=0.10, help="Weight lambda_adv for Adversarial loss (default: 0.10)")
    loss_group.add_argument("--lambda_patch", type=float, default=1.0, help="Weight lambda_patch for InfoNCE loss (default: 1.0)")
    loss_group.add_argument("--lambda_l1", type=float, default=0.20, help="Weight lambda_l1 for auxiliary L1 Pixel loss (default: 0.20)")
    loss_group.add_argument("--lambda_edge", type=float, default=0.0, help="Weight lambda_edge for auxiliary Sobel Edge loss (default: 0.0)")
    loss_group.add_argument("--temperature", "--tau", type=float, default=0.07, help="InfoNCE temperature tau (default: 0.07)")
    loss_group.add_argument("--num_patches", type=int, default=256, help="Sampled patches per layer (default: 256)")

    # 5. Evaluation & Anomaly Scoring
    eval_group = parser.add_argument_group("Evaluation & Anomaly Scoring")
    eval_group.add_argument("--reduction", type=str, default="top_percent", choices=["max", "top_percent", "top_k", "mean"], help="Anomaly reduction method (default: top_percent)")
    eval_group.add_argument("--top_k_ratio", type=float, default=0.10, help="Top percentage ratio (default: 0.10)")
    eval_group.add_argument("--top_k", type=int, default=100, help="Top pixels for top_k (default: 100)")
    eval_group.add_argument("--use_projection_head", action="store_true", help="Use Projection Head P_l during evaluation (default: False, direct cosine is more robust)")
    eval_group.add_argument("--use_global_thresholding", action="store_true", default=False, help="Enable Global Baseline Thresholding")

    # 6. Logging & Checkpoints
    log_group = parser.add_argument_group("Logging & Checkpoints")
    log_group.add_argument("--eval_every", type=int, default=5, help="Epoch frequency for evaluation (default: 5)")
    log_group.add_argument("--save_every", type=int, default=5, help="Epoch frequency for saving checkpoints (default: 5)")
    log_group.add_argument("--log_interval", type=int, default=20, help="Batch logging interval (default: 20)")
    log_group.add_argument("--checkpoint_dir", type=str, default="PatchCL_AE_MHSA/checkpoints", help="Directory to save checkpoints")
    log_group.add_argument("--log_dir", type=str, default="PatchCL_AE_MHSA/logs", help="Directory to save logs")

    args = parser.parse_args()

    use_amp = not args.no_amp and torch.cuda.is_available()
    amp_status_str = "ON / True (Enabled)" if use_amp else "OFF / False"

    print("=" * 65, flush=True)
    print("PATCHCL-AE WITH INTEGRATED MHSA - TRAINING INITIALIZATION", flush=True)
    print(f"  - Dataset:           {args.data_root}", flush=True)
    print(f"  - MHSA Mode:         {args.mhsa_mode} (heads={args.num_heads}, gamma={args.gamma})", flush=True)
    print(f"  - Feature Layers:    {args.feature_layers}", flush=True)
    print(f"  - Crop Size:         {args.resize_size}x{args.resize_size} -> {args.img_size}x{args.img_size}", flush=True)
    print(f"  - Loss Weights:      lambda_adv={args.lambda_adv}, lambda_patch={args.lambda_patch}, lambda_l1={args.lambda_l1}", flush=True)
    print(f"  - Epochs:            {args.epochs} (eval every {args.eval_every}, save every {args.save_every})", flush=True)
    print(f"  - Batch Size:        {args.batch_size}", flush=True)
    print(f"  - Learning Rates:    G={args.lr}, D={args.lr_d}", flush=True)
    print(f"  - Mixed Prec (AMP):  {amp_status_str}", flush=True)
    print(f"  - Checkpoint Dir:    {args.checkpoint_dir}", flush=True)
    print(f"  - Log Dir:           {args.log_dir}", flush=True)
    print("=" * 65, flush=True)

    loaders = create_patchcl_dataloaders(
        root_path=args.data_root,
        img_size=args.img_size,
        resize_size=args.resize_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    print(f"Train samples (Normal only): {len(loaders['train'].dataset)}", flush=True)
    print(f"Test samples (Normal + Abnormal): {len(loaders['test'].dataset)}", flush=True)

    gen, disc = create_patchcl_ae_models(
        in_channels=3,
        feature_layers=args.feature_layers,
        num_heads=args.num_heads,
        gamma=args.gamma,
        mhsa_mode=args.mhsa_mode,
        num_conv_bottlenecks=args.num_conv_bottlenecks,
    )

    total_g_params = sum(p.numel() for p in gen.parameters())
    total_d_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator parameters:     {total_g_params:,}", flush=True)
    print(f"Discriminator parameters: {total_d_params:,}", flush=True)

    trainer = PatchCLTrainer(
        generator=gen,
        discriminator=disc,
        train_loader=loaders["train"],
        test_loader=loaders["test"],
        lr=args.lr,
        lr_d=args.lr_d,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        noise_std=args.noise_std,
        temperature=args.temperature,
        num_patches=args.num_patches,
        lambda_adv=args.lambda_adv,
        lambda_patch=args.lambda_patch,
        lambda_l1=args.lambda_l1,
        lambda_edge=args.lambda_edge,
        gamma=args.gamma,
        use_global_thresholding=args.use_global_thresholding,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_amp=use_amp,
        log_interval=args.log_interval,
        eval_every=args.eval_every,
        save_every=args.save_every,
        device=args.device,
        reduction=args.reduction,
        top_k_ratio=args.top_k_ratio,
        top_k=args.top_k,
        use_projection_head=args.use_projection_head,
        feature_layers=args.feature_layers,
    )

    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
