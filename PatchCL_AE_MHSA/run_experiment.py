"""Training and Evaluation experiment runner for PatchCL-AE with MHSA on Retinal OCT.
"""

from __future__ import annotations

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
from PatchCL_AE_MHSA.visualize_oct_reconstruction import evaluate_and_visualize_oct_samples


def main() -> None:
    print("=" * 70, flush=True)
    print("=== STARTING RETINAL OCT MHSA-PATCHCL-AE TRAINING (12-15 EPOCHS) ===", flush=True)
    print("=" * 70, flush=True)

    data_root = "datasets/OCT2017"
    epochs = 12
    batch_size = 16
    max_train_samples = 2000  # 2000 normal macular scans for high-speed & high-quality training
    max_test_samples = None   # Evaluate on all 968 test samples
    lr_g = 0.0002
    lr_d = 0.000001           # 1e-6 for discriminator stability
    lambda_adv = 0.10
    lambda_patch = 1.0
    lambda_l1 = 0.25          # Crisp anatomical reconstruction
    eval_every = 4            # Evaluate at epoch 4, 8, 12
    save_every = 4
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

    # 1. Dataloaders
    print(f"\nLoading OCT2017 Dataset from '{data_root}'...", flush=True)
    loaders = create_patchcl_dataloaders(
        root_path=data_root,
        img_size=224,
        resize_size=256,
        batch_size=batch_size,
        num_workers=2,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
    print(f"Train Normal Samples: {len(loaders['train'].dataset)} ({len(loaders['train'])} batches)", flush=True)
    print(f"Test Samples:         {len(loaders['test'].dataset)} ({len(loaders['test'])} batches)", flush=True)

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

    # 3. Trainer with custom visualization callback
    class CustomPatchCLTrainer(PatchCLTrainer):
        def evaluate(self, epoch: int):
            metrics = super().evaluate(epoch)
            # Visualize reconstruction quality at this epoch
            viz_out = os.path.join(viz_dir, f"oct_reconstruction_epoch_{epoch:03d}.png")
            print(f"\n[Epoch {epoch:03d}] Generating reconstruction quality inspection figure...", flush=True)
            res_by_class = evaluate_and_visualize_oct_samples(
                model=self.generator,
                data_root=data_root,
                output_path=viz_out,
                device=self.device,
            )
            print(f"[Epoch {epoch:03d}] Class-wise scores: {res_by_class}", flush=True)
            return metrics

    trainer = CustomPatchCLTrainer(
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
        log_interval=25,
        eval_every=eval_every,
        save_every=save_every,
        device=device,
        use_projection_head=False,
        feature_layers=["E_1", "E_2", "E_3", "E_4"],
    )

    # 4. Initial pre-training reconstruction visualization
    print("\n[Sanity Check] Generating initial pre-training baseline visualization...", flush=True)
    init_viz = os.path.join(viz_dir, "oct_reconstruction_epoch_000_init.png")
    evaluate_and_visualize_oct_samples(
        model=gen,
        data_root=data_root,
        output_path=init_viz,
        device=device,
    )

    # 5. Train
    print("\nStarting Training...", flush=True)
    t_start = time.time()
    trainer.train(epochs=epochs)
    t_total = time.time() - t_start

    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETED IN {t_total/60:.2f} MINUTES!", flush=True)
    print(f"Peak AUROC:        {trainer.best_auc:.2f}%", flush=True)
    print(f"Peak F1-Score:     {trainer.best_f1:.2f}%", flush=True)
    print(f"Best Checkpoint:   {os.path.join(checkpoint_dir, 'patchcl_ae_best_auc.pt')}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
