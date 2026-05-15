from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

LOGGER = logging.getLogger(__name__)


def _to_imagenet_input(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] normalized input to ImageNet-normalized tensor."""
    x01 = torch.clamp((x + 1.0) * 0.5, min=0.0, max=1.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


class ResNetFeatureEncoder(nn.Module):
    """Feature encoder exposing one or more intermediate ResNet-50 feature maps."""

    def __init__(
        self,
        layer_name: str = "layer3",
        pretrained: bool = True,
        layer_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load pretrained ResNet-50 weights (%s). Falling back to random initialization.",
                exc,
            )
            backbone = resnet50(weights=None)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if layer_names is None:
            layer_names = [layer_name]
        elif isinstance(layer_names, str):
            layer_names = [layer_names]
        self.layer_names = list(layer_names)
        self.layer_name = self.layer_names[-1]

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_imagenet_input(x)
        x = self.stem(x)

        outputs = {}
        x1 = self.layer1(x)
        outputs["layer1"] = x1

        x2 = self.layer2(x1)
        outputs["layer2"] = x2

        x3 = self.layer3(x2)
        outputs["layer3"] = x3

        x4 = self.layer4(x3)
        outputs["layer4"] = x4

        selected = [outputs[name] for name in self.layer_names]
        target_size = selected[0].shape[-2:]
        aligned = [
            feat if feat.shape[-2:] == target_size else F.interpolate(
                feat, size=target_size, mode="bilinear", align_corners=False
            )
            for feat in selected
        ]
        return torch.cat(aligned, dim=1)


class MahalanobisFeatureModel(nn.Module):
    """Spatial PaDiM-style normal feature distribution estimator."""

    def __init__(
        self,
        layer_name: str = "layer3",
        layer_names: Optional[Sequence[str]] = None,
        pretrained: bool = True,
        diagonal_covariance: bool = True,
        covariance_eps: float = 1e-3,
        fit_max_samples: int = 20000,
    ) -> None:
        super().__init__()
        if layer_names is None:
            layer_names = [layer_name]
        elif isinstance(layer_names, str):
            layer_names = [layer_names]
        self.layer_names = list(layer_names)
        self.encoder = ResNetFeatureEncoder(
            layer_name=layer_name,
            pretrained=pretrained,
            layer_names=self.layer_names,
        )
        self.diagonal_covariance = diagonal_covariance
        self.covariance_eps = covariance_eps
        self.fit_max_samples = fit_max_samples

        self.register_buffer("mean", torch.empty(0), persistent=True)
        self.register_buffer("inv_cov", torch.empty(0), persistent=True)

    def is_stats_compatible(self, stats: Dict[str, torch.Tensor]) -> bool:
        stats_layers = stats.get("layer_names")
        if stats_layers is None:
            return False
        if isinstance(stats_layers, str):
            loaded_layers = stats_layers.split(",")
        else:
            loaded_layers = list(stats_layers)
        if loaded_layers != self.layer_names:
            return False
        return stats.get("feature_mode") == "spatial_padim"

    @torch.no_grad()
    def fit(self, train_loader, device: torch.device) -> None:
        self.encoder.eval()

        sum_feat = None
        sum_sq_feat = None
        collected = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            feat = self.encoder(images)
            if sum_feat is None:
                sum_feat = feat.sum(dim=0)
                sum_sq_feat = (feat * feat).sum(dim=0)
            else:
                sum_feat += feat.sum(dim=0)
                sum_sq_feat += (feat * feat).sum(dim=0)
            collected += feat.shape[0]
            if collected >= self.fit_max_samples:
                break

        if sum_feat is None or sum_sq_feat is None or collected == 0:
            raise RuntimeError("No features collected to fit Mahalanobis model.")

        mean = sum_feat / collected
        variance = (sum_sq_feat / collected) - mean.pow(2)
        variance = torch.clamp(variance, min=0.0) + self.covariance_eps
        inv_cov = 1.0 / variance

        self.mean = mean.to(device)
        self.inv_cov = inv_cov.to(device)

    @torch.no_grad()
    def mahalanobis_map(self, x: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        if self.mean.numel() == 0 or self.inv_cov.numel() == 0:
            raise RuntimeError("Mahalanobis stats are empty. Call fit() or load stats first.")

        feat = self.encoder(x)
        if self.mean.dim() == 3:
            diff = feat - self.mean.unsqueeze(0)
            distance = torch.sqrt(torch.sum(diff * diff * self.inv_cov.unsqueeze(0), dim=1, keepdim=True) + 1e-8)
        else:
            b, c, h, w = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            diff = feat_flat - self.mean.unsqueeze(0)
            if self.inv_cov.dim() == 1:
                distance = torch.sqrt(torch.sum(diff * diff * self.inv_cov.unsqueeze(0), dim=1) + 1e-8)
            else:
                distance = torch.sqrt(torch.sum((diff @ self.inv_cov) * diff, dim=1) + 1e-8)
            distance = distance.view(b, 1, h, w)

        score_map = distance
        if target_size is not None and score_map.shape[-2:] != target_size:
            score_map = F.interpolate(score_map, size=target_size, mode="bilinear", align_corners=False)
        score_map = torch.log1p(score_map)
        return score_map

    def get_stats(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "inv_cov": self.inv_cov.detach().cpu(),
            "diagonal_covariance": torch.tensor(int(self.diagonal_covariance), dtype=torch.int64),
            "feature_mode": "spatial_padim",
            "layer_names": self.layer_names,
        }

    def load_stats(self, stats: Dict[str, torch.Tensor], device: torch.device) -> None:
        self.mean = stats["mean"].to(device)
        self.inv_cov = stats["inv_cov"].to(device)
        if "diagonal_covariance" in stats:
            self.diagonal_covariance = bool(int(stats["diagonal_covariance"].item()))
        if "layer_names" in stats:
            self.layer_names = list(stats["layer_names"])
