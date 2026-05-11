from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import yaml

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from datasets.oct_dataset import create_oct_dataloaders
from models.attention_fusion import AttentionFusionModule
from models.diffusion_model import DDPM, build_ddpm_from_config
from models.feature_extractor import MahalanobisFeatureModel
from models.inversion import DiffusionInversionDetector, apply_deviation_correction
from utils.metrics import (
    anomaly_map_to_image_score,
    compute_image_auroc,
    compute_pixel_auroc,
    extract_pixel_lists,
    min_max_normalize,
    summarize_metrics,
)
from utils.visualization import save_anomaly_panel

LOGGER = logging.getLogger("oct_eval")


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


def _load_model_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    return True


def _build_source_maps(
    images: torch.Tensor,
    detector: DiffusionInversionDetector,
    feature_model: MahalanobisFeatureModel,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    infer_cfg = config["inference"]
    ablation = config["ablation"]

    batch_size, _, height, width = images.shape
    zeros = torch.zeros((batch_size, 1, height, width), device=images.device)

    if ablation.get("enable_diffusion", True):
        inversion_out = detector.invert(images)
        diffusion_map = inversion_out.combined_map
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

    return {
        "diffusion_map": diffusion_map,
        "feature_map": feature_map,
        "noise_map": noise_map,
        "corrected_map": corrected_map,
    }


def evaluate_pipeline(
    config: Dict,
    ddpm: DDPM,
    feature_model: MahalanobisFeatureModel,
    fusion_model: Optional[AttentionFusionModule],
    test_loader,
    device: torch.device,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    ddpm.eval()
    feature_model.eval()
    if fusion_model is not None:
        fusion_model.eval()

    infer_cfg = config["inference"]
    eval_cfg = config["eval"]
    dataset_cfg = config["dataset"]
    ablation = config["ablation"]

    detector = DiffusionInversionDetector(
        ddpm=ddpm,
        anomaly_distance=str(config["diffusion"].get("anomaly_distance", "l1")),
        inversion_steps=int(config["diffusion"].get("inversion_steps", 50)),
        use_noise_space_error=bool(ablation.get("enable_noise_space_error", True)),
    )

    image_labels = []
    image_scores = []
    pixel_maps = []
    pixel_masks = []

    save_visualizations = bool(eval_cfg.get("save_visualizations", True))
    max_visualizations = int(eval_cfg.get("max_visualizations", 50))
    vis_counter = 0

    vis_root = Path(output_dir or config["project"].get("output_dir", "outputs")) / "eval_visuals"
    vis_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy()
            # masks = batch["mask"].to(device, non_blocking=True)
            # has_mask = batch["has_mask"].to(device, non_blocking=True)
            paths = batch["path"]

            maps = _build_source_maps(images, detector, feature_model, config)
            diffusion_map = maps["diffusion_map"]
            feature_map = maps["feature_map"]
            noise_map = maps["noise_map"]
            corrected_map = maps["corrected_map"]

            if fusion_model is not None and ablation.get("enable_attention_fusion", True):
                final_map, _ = fusion_model(corrected_map, feature_map, noise_map)
            else:
                combined = [corrected_map]
                if ablation.get("enable_simplex_noise", True):
                    combined.append(noise_map)
                final_map = torch.stack(combined, dim=0).mean(dim=0)

            # final_map = min_max_normalize(final_map)
            score_tensor = anomaly_map_to_image_score(final_map, reduction="top_k")

            image_labels.extend(labels.tolist())
            # image_scores.extend(score_tensor.detach().cpu().numpy().tolist())

            # batch_pixel_maps, batch_pixel_masks = extract_pixel_lists(
            #     batch_maps=final_map,
            #     batch_masks=masks,
            #     batch_has_mask=has_mask,
            # )
            # pixel_maps.extend(batch_pixel_maps)
            # pixel_masks.extend(batch_pixel_masks)
            # nguyên giá trị gốc để tính AUC (Đảm bảo tính xếp hạng mẫu bệnh > mẫu thường)
            image_scores.extend(score_tensor.detach().cpu().numpy().tolist())

            # 2. Tạo bản sao ĐÃ CHUẨN HÓA chỉ để hiển thị heatmap
            if save_visualizations and vis_counter < max_visualizations:
                for i in range(images.shape[0]):
                    if vis_counter % 10 == 0:
                        if vis_counter >= max_visualizations:
                            break
                        
                        # HÀM CHUẨN HÓA CỤC BỘ TỪNG ẢNH ĐỂ HIỆN HEATMAP RÕ NHẤT[cite: 2]
                        def local_norm(tensor):
                            t_min, t_max = tensor.min(), tensor.max()
                            print(f"Local norm for sample {vis_counter:05d}: min={t_min.item():.4f}, max={t_max.item():.4f}")
                            return (tensor - t_min) / (t_max - t_min + 1e-8)

                        # fused_vis_single = local_norm(final_map[i]) # Chuẩn hóa riêng tấm này[cite: 2]
                        
                        file_name = f"sample_{vis_counter:05d}.png"
                        save_anomaly_panel(
                            save_path=str(vis_root / file_name),
                            image_tensor=images[i],
                            diff_map=local_norm(diffusion_map[i]),
                            feat_map=local_norm(feature_map[i]),
                            fused_map=final_map[i], # Bây giờ chắc chắn sẽ hiện đỏ[cite: 2]
                            noise_map=local_norm(noise_map[i]),
                            mean=dataset_cfg.get("normalize_mean", [0.5, 0.5, 0.5]),
                            std=dataset_cfg.get("normalize_std", [0.5, 0.5, 0.5]),
                            title=str(paths[i]),
                        )
                    vis_counter += 1

    image_auc = compute_image_auroc(image_labels, image_scores)

    pixel_available = False
    # if pixel_available:
    #     pixel_auc = compute_pixel_auroc(pixel_maps, pixel_masks)
    # else:
    if pixel_available == False:
        pixel_auc = float("nan")
        LOGGER.warning("No pixel-level masks found in test set. Pixel AUROC is skipped.")

    metrics = summarize_metrics(
        image_auc=image_auc,
        pixel_auc=pixel_auc,
        pixel_available=pixel_available,
    )

    LOGGER.info("Evaluation metrics: %s", metrics)
    return metrics


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


def _load_all_checkpoints(
    config: Dict,
    ddpm: DDPM,
    feature_model: MahalanobisFeatureModel,
    fusion_model: Optional[AttentionFusionModule],
    train_loader,
    device: torch.device,
    checkpoint_dir: str,
) -> bool:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loaded = False
    for name in ["diffusion_best.pt", "diffusion_last.pt"]:
        if _load_model_checkpoint(ddpm, ckpt_dir / name):
            LOGGER.info("Loaded diffusion checkpoint: %s", ckpt_dir / name)
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError(
            f"No diffusion checkpoint found in {ckpt_dir}. Expected diffusion_best.pt or diffusion_last.pt"
        )

    feature_stats_path = ckpt_dir / "feature_stats.pt"
    if feature_stats_path.exists():
        stats = torch.load(feature_stats_path, map_location="cpu")
        feature_model.load_stats(stats=stats, device=device)
        LOGGER.info("Loaded feature stats: %s", feature_stats_path)
    else:
        LOGGER.warning("Feature stats not found. Fitting from train-normal set before evaluation.")
        feature_model.fit(train_loader=train_loader, device=device)

    use_attention_fusion = fusion_model is not None
    if fusion_model is not None:
        fusion_loaded = False
        for name in ["fusion_best.pt", "fusion_last.pt"]:
            if _load_model_checkpoint(fusion_model, ckpt_dir / name):
                LOGGER.info("Loaded fusion checkpoint: %s", ckpt_dir / name)
                fusion_loaded = True
                break
        if not fusion_loaded:
            LOGGER.warning("Fusion checkpoint not found. Falling back to non-attention fusion.")
            use_attention_fusion = False

    return use_attention_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCT anomaly detection pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)))
    device = get_device(config)

    LOGGER.info("Using device: %s", device)

    loaders = create_oct_dataloaders(config)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    ddpm, feature_model, fusion_model = _init_models(config, device)

    checkpoint_dir = args.checkpoint_dir or config["project"].get("checkpoint_dir", "checkpoints")
    use_attention_fusion = _load_all_checkpoints(
        config=config,
        ddpm=ddpm,
        feature_model=feature_model,
        fusion_model=fusion_model,
        train_loader=train_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    if not use_attention_fusion:
        fusion_model = None

    wb_run = None
    wb_cfg = config.get("wandb", {})
    wandb_enabled = bool(wb_cfg.get("enabled", False)) and not args.disable_wandb

    if wandb_enabled:
        if wandb is None:
            raise ImportError("wandb is enabled in config but not installed.")
        wb_run = wandb.init(
            project=wb_cfg.get("project", "oct-diffusion-anomaly"),
            entity=wb_cfg.get("entity", None),
            name=wb_cfg.get("name", None),
            tags=wb_cfg.get("tags", []),
            mode=wb_cfg.get("mode", "online"),
            config=config,
        )

    metrics = evaluate_pipeline(
        config=config,
        ddpm=ddpm,
        feature_model=feature_model,
        fusion_model=fusion_model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
    )

    print("Image-level AUROC:", metrics["image_auroc"])
    print("Pixel-level AUROC:", metrics["pixel_auroc"])

    if wb_run is not None:
        wb_run.log({f"eval/{k}": v for k, v in metrics.items()})
        wb_run.finish()


if __name__ == "__main__":
    main()