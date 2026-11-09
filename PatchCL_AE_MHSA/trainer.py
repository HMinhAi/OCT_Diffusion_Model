"""Trainer for Decoupled Dual-Stream PatchCL-AE.

Features:
    - Direct joint training from Epoch 1 (no warm-up, no k-means, no memory bank overhead).
    - Decoupled parallel guidance stream for InfoNCE penalty attenuation.
    - Automatic Mixed Precision (AMP) to accelerate training and reduce VRAM usage.
    - Robust checkpoint management (latest, best AUC, best F1).
    - Evaluation with Quantile Soft-Thresholding background noise suppression.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .anomaly_score import (
    anomaly_map_to_score,
    compute_global_baseline_thresholds,
    compute_patchcl_anomaly_map,
)
from .logger import TrainingLogger
from .losses import PatchCL_TotalLoss
from .metrics import calculate_anomaly_metrics
from .models import Discriminator, PatchCL_AE

LOGGER = logging.getLogger(__name__)


class PatchCLTrainer:
    """Trainer class encapsulating complete training and evaluation workflow."""

    def __init__(
        self,
        generator: PatchCL_AE,
        discriminator: Discriminator,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 0.002,
        lr_d: Optional[float] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-4,
        noise_std: float = 0.05,
        temperature: float = 0.07,
        num_patches: int = 256,
        lambda_adv: float = 1.0,
        lambda_patch: float = 1.0,
        lambda_l1: float = 0.0,
        lambda_edge: float = 0.0,
        gamma: float = 2.0,
        quantile_ratio: float = 0.60,
        checkpoint_dir: str = "PatchCL_AE/checkpoints",
        log_dir: str = "PatchCL_AE/logs",
        use_amp: bool = True,
        amp_dtype: Optional[Union[str, torch.dtype]] = None,
        drive_backup_dir: Optional[str] = None,
        log_interval: int = 20,
        eval_every: int = 5,
        save_every: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        reduction: str = "top_percent",
        top_k_ratio: float = 0.10,
        top_k: int = 100,
        use_global_thresholding: bool = False,
        global_threshold_quantile: float = 0.95,
        global_thresholds: Optional[Dict[str, float]] = None,
        use_projection_head: bool = False,
        feature_layers: Optional[List[str]] = None,
        layer_weights: Optional[Union[Dict[str, float], List[float]]] = None,
        **kwargs,  # Gracefully ignore deprecated memory arguments
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.lr = lr
        self.lr_d = lr if lr_d is None else lr_d
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.noise_std = noise_std
        self.temperature = temperature
        self.num_patches = num_patches
        self.lambda_adv = lambda_adv
        self.lambda_patch = lambda_patch
        self.lambda_l1 = lambda_l1
        self.lambda_edge = lambda_edge
        self.gamma = gamma
        self.quantile_ratio = quantile_ratio

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.drive_backup_dir = drive_backup_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.drive_backup_dir:
            try:
                os.makedirs(self.drive_backup_dir, exist_ok=True)
            except Exception:
                pass

        self.log_interval = log_interval
        self.eval_every = eval_every
        self.save_every = save_every
        self.reduction = reduction
        self.top_k_ratio = top_k_ratio
        self.top_k = top_k
        self.use_global_thresholding = use_global_thresholding
        self.global_threshold_quantile = global_threshold_quantile
        self.global_thresholds = global_thresholds
        self.use_projection_head = use_projection_head
        self.feature_layers = feature_layers or getattr(self.generator, "feature_layers", ["E_1", "E_2", "E_3", "E_4"])
        self.layer_weights = layer_weights

        # Optimizers (Algorithm 1)
        self.optimizer_g = Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
        )
        self.optimizer_d = Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
        )

        # Loss Criterion
        self.criterion = PatchCL_TotalLoss(
            temperature=self.temperature,
            num_patches=self.num_patches,
            lambda_patch=self.lambda_patch,
            lambda_adv=self.lambda_adv,
            lambda_l1=self.lambda_l1,
            lambda_edge=self.lambda_edge,
            gamma=self.gamma,
        )

        # AMP Mixed Precision Scaler & Dtype
        self.use_amp = use_amp and (self.device.type == "cuda")
        if isinstance(amp_dtype, str):
            self.amp_dtype = torch.bfloat16 if amp_dtype.lower() == "bfloat16" else torch.float16
        elif isinstance(amp_dtype, torch.dtype):
            self.amp_dtype = amp_dtype
        else:
            self.amp_dtype = torch.float16

        # Loss scaler: only float16 requires GradScaler, bfloat16 does not
        need_scaler = self.use_amp and (self.amp_dtype == torch.float16)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=need_scaler)

        self.logger = TrainingLogger(log_dir=self.log_dir)

        # Tracking state
        self.best_auc = 0.0
        self.best_f1 = 0.0
        self.current_epoch = 0

    def autocast(self):
        """Context manager for automatic mixed precision across PyTorch versions."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input image (Paper Eq. 1: X_hat = De(E(X + Noise)))."""
        if self.noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x + noise, min=-1.0, max=1.0)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train model for one epoch using Decoupled Dual-Stream PatchCL-AE."""
        self.generator.train()
        self.discriminator.train()

        running_loss_g = 0.0
        running_loss_d = 0.0
        running_loss_adv = 0.0
        running_loss_patch = 0.0
        running_loss_l1 = 0.0
        total_batches = len(self.train_loader)

        pbar = tqdm(
            enumerate(self.train_loader),
            total=total_batches,
            desc=f"Epoch {epoch:03d} [Train]",
            leave=False,
            file=sys.stdout,
        )

        for batch_idx, batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            noisy_images = self.add_noise(images)

            # ==================================================================
            # 1. Update Generator (Encoder + Decoder + Projection + Parallel MHSA)
            # ==================================================================
            self.optimizer_g.zero_grad(set_to_none=True)

            with self.autocast():
                # Forward noisy images through MHSA-enhanced PatchCL-AE
                recon_images, feats_noisy, w_attenuation = self.generator(
                    noisy_images, return_features=True
                )

                # Extract multi-layer features from clean input X and reconstructed X_hat
                if hasattr(self.generator, "encode"):
                    _, feats_clean = self.generator.encode(images)
                    _, feats_recon = self.generator.encode(recon_images)
                else:
                    _, feats_clean = self.generator.encoder(images, return_features=True)
                    _, feats_recon = self.generator.encoder(recon_images, return_features=True)

                # Evaluate through Discriminator
                d_fake_logits = self.discriminator(recon_images)

                # Compute Generator Total Loss
                loss_g, loss_adv, loss_patch, loss_l1, loss_edge = self.criterion.forward_generator(
                    d_fake_logits=d_fake_logits,
                    features_real=feats_clean,
                    features_fake=feats_recon,
                    projection_module=self.generator.projection,
                    w_attenuation=w_attenuation,
                    layers=self.feature_layers,
                    real_images=images,
                    recon_images=recon_images,
                    return_all=True,
                )

                # Parallel MHSA self-reconstruction objective (learn normal anatomy without backprop to AE)
                if isinstance(feats_noisy, dict) and "recon_feat_mhsa" in feats_noisy and feats_noisy["recon_feat_mhsa"] is not None:
                    loss_mhsa = F.mse_loss(feats_noisy["recon_feat_mhsa"], feats_noisy["E_4"].detach())
                    loss_g = loss_g + 0.10 * loss_mhsa

            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.optimizer_g)

            # ==================================================================
            # 2. Update Discriminator D (Paper Algorithm 1, Step 6)
            # ==================================================================
            self.optimizer_d.zero_grad(set_to_none=True)

            with self.autocast():
                d_real_logits = self.discriminator(images)
                d_fake_logits_d = self.discriminator(recon_images.detach())

                loss_d, loss_real, loss_fake = self.criterion.forward_discriminator(
                    d_real_logits=d_real_logits,
                    d_fake_logits=d_fake_logits_d,
                )

            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.optimizer_d)
            self.scaler.update()

            # Record metrics
            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item()
            running_loss_adv += loss_adv.item()
            running_loss_patch += loss_patch.item()
            running_loss_l1 += loss_l1.item()

            pbar.set_postfix({
                "L_D": f"{loss_d.item():.4f}",
                "L_G": f"{loss_g.item():.4f}",
                "Adv": f"{loss_adv.item():.4f}",
                "Patch": f"{loss_patch.item():.4f}",
                "L1": f"{loss_l1.item():.4f}",
            })

            if (batch_idx + 1) % self.log_interval == 0 or (batch_idx + 1) == total_batches:
                print(
                    f"Epoch [{epoch:03d}] Batch [{batch_idx+1:04d}/{total_batches:04d}] "
                    f"Loss_G: {loss_g.item():.4f} (Adv: {loss_adv.item():.4f}, Patch: {loss_patch.item():.4f}, L1: {loss_l1.item():.4f}) | "
                    f"Loss_D: {loss_d.item():.4f}",
                    flush=True,
                )

        pbar.close()

        epoch_metrics = {
            "loss_g": running_loss_g / max(1, total_batches),
            "loss_d": running_loss_d / max(1, total_batches),
            "loss_adv": running_loss_adv / max(1, total_batches),
            "loss_patch": running_loss_patch / max(1, total_batches),
            "loss_l1": running_loss_l1 / max(1, total_batches),
        }
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """Evaluate model on testing set to differentiate normal and abnormal samples."""
        self.generator.eval()

        if self.use_global_thresholding:
            # Ngưỡng tau_l được tính cố định 1 lần duy nhất trên toàn tập Validation Normal (train_loader normal-only)
            self.global_thresholds = compute_global_baseline_thresholds(
                model=self.generator,
                val_loader=self.train_loader,
                quantile=self.global_threshold_quantile,
                device=self.device,
                max_samples=100,
            )
            print(
                f"[Eval Epoch {epoch:03d}] Global Thresholds (tau_l^global, q={self.global_threshold_quantile}): "
                f"{ {k: round(v, 4) for k, v in self.global_thresholds.items()} }",
                flush=True,
            )

        all_scores = []
        all_scores_res = []
        all_labels = []

        pbar = tqdm(
            self.test_loader,
            desc=f"Epoch {epoch:03d} [Eval]",
            leave=False,
            file=sys.stdout,
        )

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].cpu().numpy()

            with self.autocast():
                # 1. Feature discrepancy
                anomaly_maps, x_rec, _ = compute_patchcl_anomaly_map(
                    model=self.generator,
                    x=images,
                    layers=self.feature_layers,
                    layer_weights=self.layer_weights,
                    top_k_ratio=self.top_k_ratio,
                    use_thresholding=False,
                    use_projection_head=self.use_projection_head,
                )

                # Giảm bản đồ về điểm bất thường mức ảnh bằng trung bình top 10% pixel
                scores = anomaly_map_to_score(
                    anomaly_maps,
                    reduction=self.reduction,
                    top_k_ratio=self.top_k_ratio,
                    top_k=self.top_k,
                )

                # 2. Pixel reconstruction residual |X - X_hat| (top 10% after 10px border crop)
                diff = torch.abs(images - x_rec).mean(dim=1, keepdim=True)
                if diff.shape[-2] > 20 and diff.shape[-1] > 20:
                    diff_crop = diff[:, :, 10:-10, 10:-10]
                else:
                    diff_crop = diff
                flat_diff = diff_crop.reshape(diff_crop.shape[0], -1)
                k_res = max(1, int(self.top_k_ratio * flat_diff.size(1)))
                scores_res = torch.topk(flat_diff, k=k_res, dim=1).values.mean(dim=1)

            all_scores.extend(scores.cpu().float().numpy().tolist())
            all_scores_res.extend(scores_res.cpu().float().numpy().tolist())
            all_labels.extend(labels.tolist())

        pbar.close()

        # Compute AUC, F1, ACC, SEN, SPE for both methods
        metrics = calculate_anomaly_metrics(all_labels, all_scores)
        metrics_res = calculate_anomaly_metrics(all_labels, all_scores_res)

        print("=" * 60, flush=True)
        print(f"EVALUATION SUMMARY [Feature Discrepancy] (Epoch {epoch:03d}):", flush=True)
        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)
        if (labels_arr == 0).any():
            print(f"  - Normal Scores:   min={scores_arr[labels_arr==0].min():.4f}, mean={scores_arr[labels_arr==0].mean():.4f}, max={scores_arr[labels_arr==0].max():.4f}", flush=True)
        if (labels_arr == 1).any():
            print(f"  - Abnormal Scores: min={scores_arr[labels_arr==1].min():.4f}, mean={scores_arr[labels_arr==1].mean():.4f}, max={scores_arr[labels_arr==1].max():.4f}", flush=True)
        print(f"  - AUC:             {metrics['auc']:.2f}% | F1: {metrics['f1']:.2f}% | ACC: {metrics['acc']:.2f}%", flush=True)
        print(f"  - Sensitivity:     {metrics['sen']:.2f}% | Specificity: {metrics['spe']:.2f}%", flush=True)
        if 'youden_sen' in metrics:
            print(f"  - Balanced (Youden): SEN={metrics['youden_sen']:.2f}%, SPE={metrics['youden_spe']:.2f}%, ACC={metrics['youden_acc']:.2f}%", flush=True)
        if 'sen_at_spe95' in metrics:
            print(f"  - At SPE >= 95.0%:   SEN={metrics['sen_at_spe95']:.2f}%, SPE={metrics['spe_at_spe95']:.2f}%", flush=True)
        if 'sen_at_spe90' in metrics:
            print(f"  - At SPE >= 90.0%:   SEN={metrics['sen_at_spe90']:.2f}%, SPE={metrics['spe_at_spe90']:.2f}%", flush=True)
        if 'sen_at_spe85' in metrics:
            print(f"  - At SPE >= 85.0%:   SEN={metrics['sen_at_spe85']:.2f}%, SPE={metrics['spe_at_spe85']:.2f}%", flush=True)
        if 'spe_at_sen93' in metrics:
            print(f"  - At SEN >= 93.0%:   SEN={metrics['sen_at_sen93']:.2f}%, SPE={metrics['spe_at_sen93']:.2f}%", flush=True)
        if metrics.get('target_met'):
            print(f"  *** GOAL REACHED! SEN >= 93% AND SPE >= 85%: SEN={metrics['target_sen']:.2f}%, SPE={metrics['target_spe']:.2f}% ***", flush=True)
        print("-" * 60, flush=True)
        print(f"EVALUATION SUMMARY [Pixel Residual |X - X_hat|] (Epoch {epoch:03d}):", flush=True)
        print(f"  - AUC:             {metrics_res['auc']:.2f}% | F1: {metrics_res['f1']:.2f}% | ACC: {metrics_res['acc']:.2f}%", flush=True)
        print(f"  - Sensitivity:     {metrics_res['sen']:.2f}% | Specificity: {metrics_res['spe']:.2f}%", flush=True)
        if 'youden_sen' in metrics_res:
            print(f"  - Balanced (Youden): SEN={metrics_res['youden_sen']:.2f}%, SPE={metrics_res['youden_spe']:.2f}%, ACC={metrics_res['youden_acc']:.2f}%", flush=True)
        if 'sen_at_spe95' in metrics_res:
            print(f"  - At SPE >= 95.0%:   SEN={metrics_res['sen_at_spe95']:.2f}%, SPE={metrics_res['spe_at_spe95']:.2f}%", flush=True)
        if 'sen_at_spe90' in metrics_res:
            print(f"  - At SPE >= 90.0%:   SEN={metrics_res['sen_at_spe90']:.2f}%, SPE={metrics_res['spe_at_spe90']:.2f}%", flush=True)
        print("=" * 60, flush=True)

        for k, v in metrics_res.items():
            metrics[f"res_{k}"] = v
        print("=" * 60, flush=True)

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        tag: str = "latest",
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Save training checkpoint to disk."""
        path = os.path.join(self.checkpoint_dir, f"patchcl_ae_{tag}.pt")
        checkpoint = {
            "epoch": epoch,
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "optimizer_g_state": self.optimizer_g.state_dict(),
            "optimizer_d_state": self.optimizer_d.state_dict(),
            "scaler_state": self.scaler.state_dict() if (self.use_amp and self.amp_dtype == torch.float16) else None,
            "best_auc": self.best_auc,
            "best_f1": self.best_f1,
            "metrics": metrics,
            "global_thresholds": self.global_thresholds,
        }
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Successfully saved checkpoint to {path}", flush=True)

        if self.drive_backup_dir:
            try:
                os.makedirs(self.drive_backup_dir, exist_ok=True)
                drive_p = os.path.join(self.drive_backup_dir, f"patchcl_ae_{tag}.pt")
                shutil.copy2(path, drive_p)
                print(f"[Drive Sync] Checkpoint backed up to: {drive_p}", flush=True)
            except Exception as e:
                print(f"[Drive Sync Warning] Could not back up to Drive ({e})", flush=True)

        return path

    def fit(
        self,
        epochs: int,
        eval_every: Optional[int] = None,
        save_every: Optional[int] = None,
        reduction: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Alias for train() with flexible parameter overrides."""
        if eval_every is not None:
            self.eval_every = eval_every
        if save_every is not None:
            self.save_every = save_every
        if reduction is not None:
            self.reduction = reduction
        if top_k is not None:
            self.top_k = top_k
        return self.train(epochs=epochs)

    def resume_checkpoint(self, checkpoint_path: str) -> int:
        """Load state from checkpoint to resume training."""
        if not os.path.exists(checkpoint_path):
            print(f"[Resume Warning] Checkpoint {checkpoint_path} not found. Starting from scratch.", flush=True)
            return 1
        print(f"[Resume] Loading checkpoint from {checkpoint_path}...", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt["generator_state"])
        self.discriminator.load_state_dict(ckpt["discriminator_state"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g_state"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d_state"])
        if self.use_amp and ckpt.get("scaler_state") and hasattr(self, "scaler"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_auc = ckpt.get("best_auc", 0.0)
        self.best_f1 = ckpt.get("best_f1", 0.0)
        self.global_thresholds = ckpt.get("global_thresholds", None)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Resume] Resumed successfully at start of Epoch {start_epoch} (Best AUC: {self.best_auc:.2f}%)", flush=True)
        return start_epoch

    def train(self, epochs: int, start_epoch: int = 1) -> Dict[str, float]:
        """Execute complete training workflow across all epochs with Cosine Annealing LR."""
        print("=" * 60, flush=True)
        print(f"Starting Training for {epochs} epochs (from epoch {start_epoch}) on device: {self.device} (AMP: {self.use_amp})", flush=True)
        print("Architecture: Decoupled Dual-Stream PatchCL-AE (Lossy Conv Bottleneck + Parallel MHSA)", flush=True)
        print("=" * 60, flush=True)

        start_time = time.time()
        
        # Schedulers to gradually cool down learning rate and prevent over-reconstruction
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=epochs, eta_min=1e-5, last_epoch=start_epoch - 2 if start_epoch > 1 else -1
        )
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=epochs, eta_min=1e-7, last_epoch=start_epoch - 2 if start_epoch > 1 else -1
        )

        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train epoch
            train_metrics = self.train_epoch(epoch)
            epoch_duration = time.time() - epoch_start
            
            # Step schedulers
            scheduler_g.step()
            scheduler_d.step()
            current_lr = float(self.optimizer_g.param_groups[0]["lr"])

            eval_metrics = None
            if (epoch % self.eval_every == 0) or (epoch == epochs):
                eval_metrics = self.evaluate(epoch)

                # Save best AUC
                if eval_metrics["auc"] > self.best_auc:
                    self.best_auc = eval_metrics["auc"]
                    self.save_checkpoint(epoch, tag="best_auc", metrics=eval_metrics)

                # Save best F1
                if eval_metrics["f1"] > self.best_f1:
                    self.best_f1 = eval_metrics["f1"]
                    self.save_checkpoint(epoch, tag="best_f1", metrics=eval_metrics)

                # Save if target (SEN >= 93% and SPE >= 85%) is met
                if eval_metrics.get("target_met", False):
                    print(f"!!! TARGET ACHIEVED AT EPOCH {epoch:03d}: SEN={eval_metrics['target_sen']:.2f}%, SPE={eval_metrics['target_spe']:.2f}% !!!", flush=True)
                    self.save_checkpoint(epoch, tag="target_spe85_sen93", metrics=eval_metrics)

            # Update best tracking in logger if evaluation occurred
            if eval_metrics is not None:
                if "auc" in eval_metrics:
                    self.logger.update_best("auc", eval_metrics["auc"], epoch, higher_is_better=True)
                if "f1" in eval_metrics:
                    self.logger.update_best("f1", eval_metrics["f1"], epoch, higher_is_better=True)

            # Log to file and console
            self.logger.log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=eval_metrics,
                lr=current_lr,
                epoch_time=epoch_duration,
            )

            # Save latest checkpoint after every single epoch
            self.save_checkpoint(epoch, tag="latest", metrics=eval_metrics)

            # Periodic saving
            if epoch % self.save_every == 0 and epoch != epochs:
                self.save_checkpoint(epoch, tag=f"epoch_{epoch:03d}", metrics=eval_metrics)

            # Clean memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        total_duration = time.time() - start_time
        print(f"\n[Finished] Training complete in {total_duration/60:.2f} minutes.", flush=True)
        print(f"  - Peak AUC achieved: {self.best_auc:.2f}%", flush=True)
        print(f"  - Peak F1 achieved:  {self.best_f1:.2f}%", flush=True)

        return {
            "best_auc": self.best_auc,
            "best_f1": self.best_f1,
            "total_duration": total_duration,
        }
