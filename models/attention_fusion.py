from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class AttentionFusionModule(nn.Module):
    """Attention-guided fusion for multi-source anomaly maps."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

    def forward(
        self,
        a_diff: torch.Tensor,
        a_feat: torch.Tensor,
        a_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = torch.cat([a_diff, a_feat, a_noise], dim=1)
        logits = self.net(x)
        weights = torch.softmax(logits, dim=1)

        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        gamma = weights[:, 2:3]

        fused = alpha * a_diff + beta * a_feat + gamma * a_noise
        return fused, {"alpha": alpha, "beta": beta, "gamma": gamma}