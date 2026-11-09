"""Parallel Scheme Training & Evaluation Script for Decoupled MHSA PatchCL-AE on Chest X-Ray.

Architecture (Approved Parallel Scheme):
    Reconstruction Stream: E_4 -> E_5 Conv (6 blocks) -> Decoder -> X_hat
        (Local lossy compression trap, sharp normal anatomy, erases disease)
    Guidance Stream:       E_4 -> Parallel MHSA -> S -> w(i, j) = (1 - S)^gamma -> InfoNCE
        (Detached, modulates InfoNCE negative penalties, zero gradient into AE)
"""

from __future__ import annotations

import json
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from PatchCL_AE_MHSA.dataset import create_patchcl_dataloaders
from PatchCL_AE_MHSA.models import create_patchcl_ae_models
from PatchCL_AE_MHSA.trainer import PatchCLTrainer
from PatchCL_AE_MHSA.visualize_chest_reconstruction import evaluate_and_visualize_chest_samples


def main() -> None:
    print("=" * 76, flush=True)
    print("=== STARTING CHEST X-RAY TRAINING: PARALLEL MHSA SCHEME (15 EPOCHS) ===", flush=True)
    print("=" * 76, flush=True)

    data_root = "datasets/chest_xray"
    epochs = 35
    batch_size = 16
    lr_g = 0.0001             # Refinement learning rate for extended epochs
    lr_d = 0.000001           # 1e-6 keeps Discriminator stable and prevents collapse
    lambda_adv = 0.10
    lambda_patch = 1.0
    lambda_l1 = 0.20          # Crisp anatomical reconstruction
    eval_every = 2            # Evaluate frequently to capture target SEN >= 93% and SPE >= 85%
    save_every = 5
    num_heads = 4
    gamma = 2.0
    checkpoint_dir = "PatchCL_AE_MHSA/checkpoints_parallel"
    log_dir = "PatchCL_AE_MHSA/logs_parallel"
    viz_dir = "visualizations/patchcl_ae_mhsa_parallel"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Dataloaders (num_workers=0 on Windows prevents paging file error 1455)
    print(f"\nLoading Chest X-Ray Dataset from '{data_root}'...", flush=True)
    loaders = create_patchcl_dataloaders(
        root_path=data_root,
        img_size=224,
        resize_size=256,
        batch_size=batch_size,
        num_workers=0,
        max_train_samples=None,  # All 1,349 normal train images
        max_test_samples=None,   # All 625 test images (235 Normal + 390 Pneumonia)
    )
    print(f"Train Normal Images: {len(loaders['train'].dataset)} ({len(loaders['train'])} batches)", flush=True)
    print(f"Test Images:         {len(loaders['test'].dataset)} ({len(loaders['test'])} batches)", flush=True)

    # 2. Models (PARALLEL SCHEME)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    gen, disc = create_patchcl_ae_models(
        in_channels=3,
        feature_layers=["E_1", "E_2", "E_3", "E_4"],
        num_heads=num_heads,
        gamma=gamma,
        mhsa_mode="parallel",
        device=device,
    )

    total_g_params = sum(p.numel() for p in gen.parameters())
    total_d_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator params (Parallel Scheme): {total_g_params:,}", flush=True)
    print(f"Discriminator params:              {total_d_params:,}", flush=True)
    print(f"MHSA Mode:                         {gen.mhsa_mode}", flush=True)
    print(f"Reconstruction Path:               E_1..E_4 -> E_5 (6 Conv blocks) -> Decoder (Isolated from MHSA)", flush=True)
    print(f"Guidance Path:                     E_4 -> Parallel MHSA -> (1 - S)^gamma -> InfoNCE (Detached)", flush=True)

    # 3. Custom Trainer with visual inspection hook
    class ParallelChestXRayTrainer(PatchCLTrainer):
        def evaluate(self, epoch: int):
            metrics = super().evaluate(epoch)
            viz_out = os.path.normpath(os.path.abspath(os.path.join(viz_dir, f"chest_parallel_epoch_{epoch:03d}.png")))
            print(f"\n[Epoch {epoch:03d}] Generating Chest X-Ray Parallel Scheme visual reconstruction...", flush=True)
            try:
                res_summary = evaluate_and_visualize_chest_samples(
                    model=self.generator,
                    data_root=data_root,
                    output_path=viz_out,
                    device=self.device,
                )
                print(f"[Epoch {epoch:03d}] Visual reconstruction summary:", flush=True)
                for k, v in res_summary.items():
                    print(f"   * {k:22s}: L1 Error = {v['avg_l1']:.4f}, Anomaly Score = {v['avg_score']:.4f}", flush=True)
            except Exception as e:
                print(f"[Epoch {epoch:03d} Visualization Warning] Could not render visual grid ({e})", flush=True)
            return metrics

    trainer = ParallelChestXRayTrainer(
        generator=gen,
        discriminator=disc,
        train_loader=loaders["train"],
        test_loader=loaders["test"],
        lr=lr_g,
        lr_d=lr_d,
        temperature=0.07,
        num_patches=256,
        lambda_adv=lambda_adv,
        lambda_patch=lambda_patch,
        lambda_l1=lambda_l1,
        gamma=gamma,
        use_global_thresholding=False,
        reduction="top_percent",
        top_k_ratio=0.10,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        use_amp=True,
        log_interval=20,
        eval_every=eval_every,
        save_every=save_every,
        device=device,
        use_projection_head=False,
        feature_layers=["E_1", "E_2", "E_3", "E_4"],
        layer_weights=[0.00, 0.05, 0.35, 0.60],
    )

    # 4. Check for existing checkpoint to resume
    start_epoch = 1
    latest_ckpt = os.path.join(checkpoint_dir, "patchcl_ae_latest.pt")
    if os.path.exists(latest_ckpt):
        start_epoch = trainer.resume_checkpoint(latest_ckpt)

    if start_epoch == 1:
        print("\n[Baseline] Generating pre-training reconstruction visualization (Epoch 000)...", flush=True)
        init_viz = os.path.normpath(os.path.abspath(os.path.join(viz_dir, "chest_parallel_epoch_000_init.png")))
        evaluate_and_visualize_chest_samples(
            model=gen,
            data_root=data_root,
            output_path=init_viz,
            device=device,
        )

    # Start Training
    print(f"\nStarting Parallel Scheme Training for {epochs} Epochs on Chest X-Ray (Resuming from Epoch {start_epoch})...", flush=True)
    t_start = time.time()
    trainer.train(epochs=epochs, start_epoch=start_epoch)
    t_total = time.time() - t_start

    # Final visual evaluation
    final_viz = os.path.normpath(os.path.abspath(os.path.join(viz_dir, "chest_parallel_final_eval.png")))
    evaluate_and_visualize_chest_samples(
        model=gen,
        data_root=data_root,
        output_path=final_viz,
        device=device,
    )

    print("\n" + "=" * 76, flush=True)
    print(f"PARALLEL SCHEME CHEST X-RAY TRAINING COMPLETED IN {t_total/60:.2f} MINUTES ({t_total:.1f}s)!", flush=True)
    print(f"Peak AUROC:        {trainer.best_auc:.2f}%", flush=True)
    print(f"Peak F1-Score:     {trainer.best_f1:.2f}%", flush=True)
    print(f"Best Checkpoint:   {os.path.join(checkpoint_dir, 'patchcl_ae_best_auc.pt')}", flush=True)
    print(f"Final Viz Grid:    {final_viz}", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main()
