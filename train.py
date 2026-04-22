from __future__ import annotations

import argparse
import copy
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from datasets.oct_dataset import create_oct_dataloaders
from eval import evaluate_pipeline
from models.attention_fusion import AttentionFusionModule
from models.diffusion_model import DDPM, build_ddpm_from_config
from models.feature_extractor import MahalanobisFeatureModel
from models.inversion import DiffusionInversionDetector, apply_deviation_correction
from utils.metrics import min_max_normalize

LOGGER = logging.getLogger("oct_train")


def detect_hardware() -> Dict:
    info: Dict = {
        "device_name": "CPU",
        "vram_gb": 0.0,
        "sm_count": 0,
        "cpu_cores": os.cpu_count() or 4,
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["vram_gb"] = props.total_memory / (1024 ** 3)
        info["sm_count"] = props.multi_processor_count
    return info


def apply_hardware_config(config: Dict, hw: Dict) -> Dict:
    config = copy.deepcopy(config)
    cpu_cores: int = hw["cpu_cores"]
    vram_gb: float = hw["vram_gb"]

    if str(config["project"].get("num_workers", 4)).lower() == "auto":
        workers = min(max(cpu_cores // 2, 2), 12)
        config["project"]["num_workers"] = workers
        LOGGER.info("Auto num_workers=%d  (cpu_cores=%d)", workers, cpu_cores)

    def _resolve_batch(key: str, section: str, default: int) -> None:
        if str(config[section].get(key, default)).lower() == "auto":
            if vram_gb >= 23:
                bs = 32
            elif vram_gb >= 14:
                bs = 16
            elif vram_gb >= 7:
                bs = 8
            else:
                bs = 4
            config[section][key] = bs
            LOGGER.info("Auto %s[%s]=%d  (vram=%.1fGB)", section, key, bs, vram_gb)

    _resolve_batch("batch_size", "train", 8)
    _resolve_batch("batch_size", "eval", 4)
    return config


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config: Dict) -> torch.device:
    requested = str(config["project"].get("device", "auto")).lower()
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _compute_maps_for_fusion_training(
    images: torch.Tensor,
    detector: DiffusionInversionDetector,
    feature_model: MahalanobisFeatureModel,
    config: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    infer_cfg = config["inference"]
    ablation = config["ablation"]

    batch_size, _, height, width = images.shape
    zeros = torch.zeros((batch_size, 1, height, width), device=images.device)

    if ablation.get("enable_diffusion", True):
        inv = detector.invert(images)
        diffusion_map = inv.combined_map
    else:
        diffusion_map = zeros

    if ablation.get("enable_feature_correction", True):
        feature_map = feature_model.mahalanobis_map(images, target_size=(height, width))
        if ablation.get("enable_diffusion", True):
            corrected_map = apply_deviation_correction(
                diffusion_map=diffusion_map,
                feature_map=feature_map,
                lambda_correction=float(infer_cfg.get("lambda_correction", 0.5)),
            )
        else:
            corrected_map = feature_map
    else:
        feature_map = zeros
        corrected_map = diffusion_map

    if ablation.get("enable_simplex_noise", True):
        noise_map = detector.simplex_robust_anomaly(
            x=images,
            runs=int(infer_cfg.get("simplex_runs", 4)),
            noise_scale=float(infer_cfg.get("simplex_noise_scale", 0.08)),
            octaves=int(infer_cfg.get("simplex_octaves", 3)),
            aggregate=str(infer_cfg.get("noise_aggregate", "mean")),
        )
    else:
        noise_map = zeros

    return corrected_map, feature_map, noise_map


def _build_pseudo_labels(
    corrected_map: torch.Tensor,
    feature_map: torch.Tensor,
    noise_map: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    corrected_norm = min_max_normalize(corrected_map)
    feature_norm = min_max_normalize(feature_map)
    noise_norm = min_max_normalize(noise_map)

    soft_map = (corrected_norm + feature_norm + noise_norm) / 3.0

    flat = soft_map.view(soft_map.shape[0], -1)
    q = max(0.0, min(1.0, float(quantile)))
    threshold = torch.quantile(flat, q=q, dim=1, keepdim=True)
    hard_map = (flat >= threshold).float().view_as(soft_map)

    # Blend hard and soft pseudo labels for stable fusion training.
    return 0.5 * soft_map + 0.5 * hard_map


def _init_wandb(config: Dict, hw: Dict, disable_wandb: bool):
    wb_cfg = config.get("wandb", {})
    enabled = bool(wb_cfg.get("enabled", False)) and not disable_wandb
    if not enabled:
        return None
    if wandb is None:
        raise ImportError("wandb is enabled in config but not installed.")

    wandb_config = copy.deepcopy(config)
    wandb_config["hardware"] = {
        "gpu_name": hw["device_name"],
        "vram_gb": round(hw["vram_gb"], 1),
        "sm_count": hw["sm_count"],
        "gpu_count": hw["gpu_count"],
        "cpu_cores": hw["cpu_cores"],
    }

    return wandb.init(
        project=wb_cfg.get("project", "oct-diffusion-anomaly"),
        entity=wb_cfg.get("entity", None),
        name=wb_cfg.get("name", None),
        tags=wb_cfg.get("tags", []),
        mode=wb_cfg.get("mode", "online"),
        config=wandb_config,
    )


def train_diffusion(
    ddpm: DDPM,
    train_loader,
    config: Dict,
    device: torch.device,
    checkpoint_dir: Path,
    wb_run,
) -> int:
    train_cfg = config["train"]
    ablation = config["ablation"]

    if not ablation.get("enable_diffusion", True):
        LOGGER.warning("Diffusion branch disabled by ablation. Skipping diffusion training.")
        save_checkpoint(checkpoint_dir / "diffusion_last.pt", {"model": ddpm.state_dict(), "epoch": 0})
        return 0

    optimizer = Adam(
        ddpm.parameters(),
        lr=float(train_cfg.get("lr_diffusion", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    epochs = int(train_cfg.get("epochs_diffusion", 100))
    log_interval = int(train_cfg.get("log_interval", 20))
    save_every = int(train_cfg.get("save_every", 5))

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        ddpm.train()
        epoch_losses = []

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss = ddpm.training_loss(images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            global_step += 1

            if wb_run is not None and (global_step % log_interval == 0):
                wb_run.log(
                    {
                        "train/diffusion_loss_step": loss_value,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        LOGGER.info("[Diffusion] Epoch %d/%d - loss=%.6f", epoch, epochs, epoch_loss)

        save_checkpoint(
            checkpoint_dir / "diffusion_last.pt",
            {"model": ddpm.state_dict(), "epoch": epoch, "loss": epoch_loss},
        )

        if epoch % save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"diffusion_epoch_{epoch:03d}.pt",
                {"model": ddpm.state_dict(), "epoch": epoch, "loss": epoch_loss},
            )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                checkpoint_dir / "diffusion_best.pt",
                {"model": ddpm.state_dict(), "epoch": epoch, "loss": best_loss},
            )

        if wb_run is not None:
            wb_run.log(
                {
                    "train/diffusion_loss_epoch": epoch_loss,
                    "train/diffusion_best_loss": best_loss,
                    "train/epoch": epoch,
                },
                step=global_step,
            )

    return global_step


def train_fusion_module(
    ddpm: DDPM,
    feature_model: MahalanobisFeatureModel,
    fusion_model: Optional[AttentionFusionModule],
    train_loader,
    config: Dict,
    device: torch.device,
    checkpoint_dir: Path,
    start_step: int,
    wb_run,
) -> int:
    if fusion_model is None:
        LOGGER.warning("Attention fusion disabled by ablation. Skipping fusion training.")
        return start_step

    train_cfg = config["train"]
    ablation = config["ablation"]

    detector = DiffusionInversionDetector(
        ddpm=ddpm,
        anomaly_distance=str(config["diffusion"].get("anomaly_distance", "l1")),
        inversion_steps=int(config["diffusion"].get("inversion_steps", 50)),
        use_noise_space_error=bool(ablation.get("enable_noise_space_error", True)),
    )

    optimizer = Adam(
        fusion_model.parameters(),
        lr=float(train_cfg.get("lr_fusion", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    epochs = int(train_cfg.get("epochs_fusion", 20))
    log_interval = int(train_cfg.get("log_interval", 20))
    pseudo_quantile = float(train_cfg.get("pseudo_label_quantile", 0.9))

    ddpm.eval()
    feature_model.eval()

    best_loss = float("inf")
    global_step = start_step

    for epoch in range(1, epochs + 1):
        fusion_model.train()
        losses = []

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)

            with torch.no_grad():
                corrected_map, feature_map, noise_map = _compute_maps_for_fusion_training(
                    images=images,
                    detector=detector,
                    feature_model=feature_model,
                    config=config,
                )

                corrected_norm = min_max_normalize(corrected_map)
                feature_norm = min_max_normalize(feature_map)
                noise_norm = min_max_normalize(noise_map)

                pseudo_target = _build_pseudo_labels(
                    corrected_map=corrected_map,
                    feature_map=feature_map,
                    noise_map=noise_map,
                    quantile=pseudo_quantile,
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                fused_map, weights = fusion_model(
                    a_diff=corrected_norm,
                    a_feat=feature_norm,
                    a_noise=noise_norm,
                )
                fused_norm = min_max_normalize(fused_map)
                loss = F.mse_loss(fused_norm, pseudo_target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.item())
            losses.append(loss_value)
            global_step += 1

            if wb_run is not None and (global_step % log_interval == 0):
                wb_run.log(
                    {
                        "train/fusion_loss_step": loss_value,
                        "train/fusion_alpha_mean": float(weights["alpha"].mean().item()),
                        "train/fusion_beta_mean": float(weights["beta"].mean().item()),
                        "train/fusion_gamma_mean": float(weights["gamma"].mean().item()),
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )

        epoch_loss = float(np.mean(losses)) if losses else float("inf")
        LOGGER.info("[Fusion] Epoch %d/%d - loss=%.6f", epoch, epochs, epoch_loss)

        save_checkpoint(
            checkpoint_dir / "fusion_last.pt",
            {"model": fusion_model.state_dict(), "epoch": epoch, "loss": epoch_loss},
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                checkpoint_dir / "fusion_best.pt",
                {"model": fusion_model.state_dict(), "epoch": epoch, "loss": best_loss},
            )

        if wb_run is not None:
            wb_run.log(
                {
                    "train/fusion_loss_epoch": epoch_loss,
                    "train/fusion_best_loss": best_loss,
                },
                step=global_step,
            )

    return global_step


def _init_models(config: Dict, device: torch.device):
    ddpm = build_ddpm_from_config(config).to(device)

    feature_cfg = config["feature"]
    feature_model = MahalanobisFeatureModel(
        layer_name=str(feature_cfg.get("layer", "layer3")),
        pretrained=bool(feature_cfg.get("pretrained", True)),
        diagonal_covariance=bool(feature_cfg.get("diagonal_covariance", True)),
        covariance_eps=float(feature_cfg.get("covariance_eps", 1e-3)),
        fit_max_samples=int(feature_cfg.get("fit_max_samples", 20000)),
    ).to(device)

    fusion_model = None
    if config["ablation"].get("enable_attention_fusion", True):
        fusion_model = AttentionFusionModule(
            hidden_channels=int(config["fusion"].get("hidden_channels", 32))
        ).to(device)

    return ddpm, feature_model, fusion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCT diffusion inversion anomaly detector.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()

    config = load_config(args.config)
    seed = int(config["project"].get("seed", 42))
    set_seed(seed)

    hw = detect_hardware()
    LOGGER.info(
        "Hardware: %s | VRAM=%.1fGB | SMs=%d | CPU cores=%d | GPUs=%d",
        hw["device_name"], hw["vram_gb"], hw["sm_count"], hw["cpu_cores"], hw["gpu_count"],
    )
    config = apply_hardware_config(config, hw)

    device = get_device(config)
    LOGGER.info("Using device: %s", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(config["project"].get("output_dir", "outputs"))
    checkpoint_dir = Path(config["project"].get("checkpoint_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(checkpoint_dir / "run_config.pt", {"config": config})

    wb_run = _init_wandb(config, hw=hw, disable_wandb=args.disable_wandb)

    loaders = create_oct_dataloaders(config)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    ddpm, feature_model, fusion_model = _init_models(config, device)

    global_step = train_diffusion(
        ddpm=ddpm,
        train_loader=train_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        wb_run=wb_run,
    )

    if config["ablation"].get("enable_feature_correction", True):
        LOGGER.info("Fitting normal feature distribution for deviation correction.")
        feature_model.fit(train_loader=train_loader, device=device)
        torch.save(feature_model.get_stats(), checkpoint_dir / "feature_stats.pt")
    else:
        LOGGER.warning("Feature correction disabled by ablation.")

    global_step = train_fusion_module(
        ddpm=ddpm,
        feature_model=feature_model,
        fusion_model=fusion_model,
        train_loader=train_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        start_step=global_step,
        wb_run=wb_run,
    )

    metrics = evaluate_pipeline(
        config=config,
        ddpm=ddpm,
        feature_model=feature_model,
        fusion_model=fusion_model,
        test_loader=test_loader,
        device=device,
        output_dir=str(output_dir),
    )

    LOGGER.info("Final evaluation: %s", metrics)
    print("Training finished.")
    print("Image-level AUROC:", metrics["image_auroc"])
    print("Pixel-level AUROC:", metrics["pixel_auroc"])

    if wb_run is not None:
        wb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
        wb_run.finish()


if __name__ == "__main__":
    main()
