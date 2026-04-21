from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
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
    """Feature encoder exposing intermediate ResNet-50 feature maps."""

    def __init__(self, layer_name: str = "layer3", pretrained: bool = True) -> None:
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

        self.layer_name = layer_name

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_imagenet_input(x)
        x = self.stem(x)

        x1 = self.layer1(x)
        if self.layer_name == "layer1":
            return x1

        x2 = self.layer2(x1)
        if self.layer_name == "layer2":
            return x2

        x3 = self.layer3(x2)
        if self.layer_name == "layer3":
            return x3

        x4 = self.layer4(x3)
        return x4


class MahalanobisFeatureModel(nn.Module):
    """Normal feature distribution estimator for deviation correction."""

    def __init__(
        self,
        layer_name: str = "layer3",
        pretrained: bool = True,
        diagonal_covariance: bool = True,
        covariance_eps: float = 1e-3,
        fit_max_samples: int = 20000,
    ) -> None:
        super().__init__()
        self.encoder = ResNetFeatureEncoder(layer_name=layer_name, pretrained=pretrained)
        self.diagonal_covariance = diagonal_covariance
        self.covariance_eps = covariance_eps
        self.fit_max_samples = fit_max_samples

        self.register_buffer("mean", torch.empty(0), persistent=True)
        self.register_buffer("inv_cov", torch.empty(0), persistent=True)

    @torch.no_grad()
    def fit(self, train_loader, device: torch.device) -> None:
        self.encoder.eval()

        chunks = []
        collected = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            feat = self.encoder(images)
            b, c, h, w = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            chunks.append(feat_flat.detach().cpu())
            collected += feat_flat.shape[0]
            if collected >= self.fit_max_samples:
                break

        if not chunks:
            raise RuntimeError("No features collected to fit Mahalanobis model.")

        features = torch.cat(chunks, dim=0)
        if features.shape[0] > self.fit_max_samples:
            indices = torch.randperm(features.shape[0])[: self.fit_max_samples]
            features = features[indices]

        mean = features.mean(dim=0)
        centered = features - mean

        if self.diagonal_covariance:
            variance = centered.pow(2).mean(dim=0) + self.covariance_eps
            inv_cov = 1.0 / variance
        else:
            np_cov = np.cov(features.numpy(), rowvar=False)
            np_cov = np_cov + self.covariance_eps * np.eye(np_cov.shape[0], dtype=np.float32)
            cov = torch.from_numpy(np_cov).float()
            inv_cov = torch.linalg.pinv(cov)

        self.mean = mean.to(device)
        self.inv_cov = inv_cov.to(device)

    @torch.no_grad()
    def mahalanobis_map(self, x: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        if self.mean.numel() == 0 or self.inv_cov.numel() == 0:
            raise RuntimeError("Mahalanobis stats are empty. Call fit() or load stats first.")

        feat = self.encoder(x)
        b, c, h, w = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
        diff = feat_flat - self.mean.unsqueeze(0)

        if self.diagonal_covariance:
            distance = torch.sqrt(torch.sum(diff * diff * self.inv_cov.unsqueeze(0), dim=1) + 1e-8)
        else:
            distance = torch.sqrt(torch.sum((diff @ self.inv_cov) * diff, dim=1) + 1e-8)

        score_map = distance.view(b, 1, h, w)
        if target_size is not None and score_map.shape[-2:] != target_size:
            score_map = F.interpolate(score_map, size=target_size, mode="bilinear", align_corners=False)

        return score_map

    def get_stats(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "inv_cov": self.inv_cov.detach().cpu(),
            "diagonal_covariance": torch.tensor(int(self.diagonal_covariance), dtype=torch.int64),
        }

    def load_stats(self, stats: Dict[str, torch.Tensor], device: torch.device) -> None:
        self.mean = stats["mean"].to(device)
        self.inv_cov = stats["inv_cov"].to(device)
        if "diagonal_covariance" in stats:
            self.diagonal_covariance = bool(int(stats["diagonal_covariance"].item()))
