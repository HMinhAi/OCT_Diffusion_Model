"""Complete Training & Visual Evaluation Script for MHSA PatchCL-AE on Chest X-Ray.
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
    print("=" * 72, flush=True)
    print("=== STARTING CHEST X-RAY MHSA-PATCHCL-AE TRAINING (30 EPOCHS) ===", flush=True)
    print("=" * 72, flush=True)

    data_root = "datasets/chest_xray"
    epochs = 50
    batch_size = 16
    lr_g = 0.0002
    lr_d = 0.000001           # 1e-6 keeps Discriminator stable and prevents collapse
    lambda_adv = 0.10
    lambda_patch = 1.0
    lambda_l1 = 0.20          # Crisp anatomical reconstruction
    eval_every = 5            # Evaluate at epoch 5, 10, 15, 20, 25, 30
    save_every = 5
    num_heads = 4
    gamma = 2.0
    checkpoint_dir = "PatchCL_AE_MHSA/checkpoints"
    log_dir = "PatchCL_AE_MHSA/logs"
    viz_dir = "visualizations/patchcl_ae_mhsa"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Dataloaders (num_workers=0 on Windows prevents error 1455 paging file exhaustion)
    print(f"\nLoading Chest X-Ray Dataset from '{data_root}'...", flush=True)
    loaders = create_patchcl_dataloaders(
        root_path=data_root,
        img_size=224,
        resize_size=256,
        batch_size=batch_size,
        num_workers=0,
        max_train_samples=None,  # Use all 1,349 normal train images
        max_test_samples=None,   # Use all 625 test images (235 Normal + 390 Pneumonia)
    )
    print(f"Train Normal Images: {len(loaders['train'].dataset)} ({len(loaders['train'])} batches)", flush=True)
    print(f"Test Images:         {len(loaders['test'].dataset)} ({len(loaders['test'])} batches)", flush=True)

    # 2. Models
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    gen, disc = create_patchcl_ae_models(
        in_channels=3,
        feature_layers=["E_1", "E_2", "E_3", "E_4"],
        num_heads=num_heads,
        gamma=gamma,
        mhsa_mode="bottleneck",
        device=device,
    )

    total_g_params = sum(p.numel() for p in gen.parameters())
    total_d_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator params:     {total_g_params:,}", flush=True)
    print(f"Discriminator params: {total_d_params:,}", flush=True)

    # 3. Custom Trainer with visual inspection hook
    class ChestXRayTrainer(PatchCLTrainer):
        def evaluate(self, epoch: int):
            metrics = super().evaluate(epoch)
            # Visualize reconstruction quality at this epoch
            viz_out = os.path.normpath(os.path.abspath(os.path.join(viz_dir, f"chest_reconstruction_epoch_{epoch:03d}.png")))
            print(f"\n[Epoch {epoch:03d}] Generating Chest X-Ray visual reconstruction comparison...", flush=True)
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

    trainer = ChestXRayTrainer(
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
    )

    # 4. Check for existing checkpoint to resume
    start_epoch = 1
    latest_ckpt = os.path.join(checkpoint_dir, "patchcl_ae_latest.pt")
    if os.path.exists(latest_ckpt):
        start_epoch = trainer.resume_checkpoint(latest_ckpt)

    if start_epoch == 1:
        print("\n[Baseline] Generating pre-training reconstruction visualization (Epoch 000)...", flush=True)
        init_viz = os.path.join(viz_dir, "chest_reconstruction_epoch_000_init.png")
        evaluate_and_visualize_chest_samples(
            model=gen,
            data_root=data_root,
            output_path=init_viz,
            device=device,
        )

    # 5. Start Training
    print(f"\nStarting Training for {epochs} Epochs on Chest X-Ray (Resuming from Epoch {start_epoch})...", flush=True)
    t_start = time.time()
    trainer.train(epochs=epochs, start_epoch=start_epoch)
    t_total = time.time() - t_start

    print("\n" + "=" * 72, flush=True)
    print(f"CHEST X-RAY TRAINING COMPLETED IN {t_total/60:.2f} MINUTES ({t_total:.1f}s)!", flush=True)
    print(f"Peak AUROC:        {trainer.best_auc:.2f}%", flush=True)
    print(f"Peak F1-Score:     {trainer.best_f1:.2f}%", flush=True)
    print(f"Best Checkpoint:   {os.path.join(checkpoint_dir, 'patchcl_ae_best_auc.pt')}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
