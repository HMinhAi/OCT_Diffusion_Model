"""PatchCL-AE Architecture with Integrated Multi-Head Self-Attention (MHSA).

Reference:
    Shuai Lu, Weihang Zhang, Jia Guo, Hanruo Liu, Huiqi Li, Ningli Wang.
    "PatchCL-AE: Anomaly detection for medical images using patch-wise contrastive learning-based auto-encoder"
    Computerized Medical Imaging and Graphics 114 (2024) 102366.
    DOI: 10.1016/j.compmedimag.2024.102366

Architectural Innovation: MHSA-Enhanced PatchCL-AE
    1. MHSA Bottleneck Integration:
       E_1 -> E_4 -> MHSABottleneckBlock -> E_5 (6 Conv blocks) -> Decoder D -> X_hat.
       - The MHSA Bottleneck captures global horizontal retinal layer context and long-range
         anatomical dependencies across all 784 spatial positions (28x28 grid).
       - The 6 Conv blocks at E_5 enforce local lossy compression to suppress anomalous lesions
         (such as CNV membranes, DME fluid cysts, and Drusen deposits).
       - Every parameter of the MHSA module receives direct end-to-end gradients from the
         reconstruction and adversarial objectives.
    2. Attention-Guided InfoNCE Negative Attenuation:
       The attention correlation matrix S from the trained MHSA modulates the InfoNCE
       contrastive loss via:
           w(i, j) = (1 - S_{i, j})^gamma
       eliminating Bilateral False Negative penalties between symmetric normal retinal layers.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# 1. Basic Building Blocks
# ==============================================================================

class ConvBlock(nn.Module):
    """Convolution + InstanceNorm + Activation block.
    
    Paper Section 3.1.1:
        "Each block consists of multiple 3x3 convolutional kernels with a stride of 1
        and padding of 1, followed by a pooling layer at the end. In addition, each
        convolutional layer is followed by an InstanceNorm layer and a ReLU activation layer by default."
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: str = "relu",
        use_norm: bool = True,
        downsample: bool = False,
        downsample_mode: str = "avgpool",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "none" or activation is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if downsample:
            if downsample_mode == "avgpool":
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
            elif downsample_mode == "maxpool":
                self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
            elif downsample_mode == "conv":
                self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            else:
                self.downsample = nn.Identity()
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class BottleneckBlock(nn.Module):
    """Plain Convolutional Block for E_5 bottleneck (Paper Table 1, Page 5).
    
    Structure:
        [conv 3x3, 256 -> InstanceNorm -> ReLU -> conv 3x3, 256 -> InstanceNorm -> ReLU] x 6
        
    Paper Table 1:
        Plain convolutional blocks without residual shortcuts. Removing residual
        connections is critical in anomaly detection to prevent identity mapping shortcuts
        that reconstruct anomalous regions too well.
    """

    def __init__(self, channels: int = 256, use_residual: bool = False) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.use_residual:
            out = self.relu2(out + x)
        else:
            out = self.relu2(out)
        return out


# Alias for backward compatibility
ResnetBlock = BottleneckBlock


# ==============================================================================
# 2. Multi-Head Self-Attention Modules
# ==============================================================================

def compute_attention_attenuation_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute affinity matrix S in [0, 1] and attenuation weights w = (1 - S)^gamma.
    
    Uses Cosine Similarity between Query and Key tokens for numerically robust affinity:
        sim_{ij} = (q_i / ||q_i||) . (k_j / ||k_j||)
        S_{sym} = 0.5 * (sim + sim^T)
        S = ReLU(S_{sym})  with diagonal set to 0.0
        w(i, j) = (1 - S)^gamma
        
    Properties:
        - At initialization: random projection gives sim ~ 0, S ~ 0, w ~ 1.0 (baseline InfoNCE preserved).
        - Learned features: mutually similar / symmetric normal regions achieve sim > 0.7 -> S -> 1 -> w -> 0.
        - Unrelated negative pairs: sim <= 0 -> S = 0 -> w = 1.0 (full contrastive penalty preserved).
        
    Args:
        q: Query tensor [B, N, C]
        k: Key tensor [B, N, C]
        attn_weights: Optional raw softmax attention weights [B, N, N]
        gamma: Exponent gamma
        
    Returns:
        w_attenuation: Attenuation factor [B, N, N]
        s_matrix: Normalized affinity matrix S [B, N, N]
    """
    b, n, _ = q.shape
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    
    cos_sim = torch.bmm(q_norm, k_norm.transpose(1, 2))  # [B, N, N] in [-1, 1]
    s_sym = 0.5 * (cos_sim + cos_sim.transpose(1, 2))     # Symmetrized
    
    # Positive affinity only (negative/orthogonal pairs get S = 0)
    s_matrix = F.relu(s_sym)
    
    # Zero out diagonal (positive pairs handled by numerator in InfoNCE)
    diag_mask = torch.eye(n, device=q.device, dtype=torch.bool).unsqueeze(0)
    s_matrix = s_matrix.masked_fill(diag_mask, 0.0)
    
    # Attenuation factor w(i, j) = (1 - S_{ij})^gamma
    s_clamped = s_matrix.clamp(min=0.0, max=1.0 - 1e-6)
    w_attenuation = torch.pow(1.0 - s_clamped, gamma)
    w_attenuation = w_attenuation.masked_fill(diag_mask, 1.0)
    
    return w_attenuation, s_matrix


class MHSABottleneckBlock(nn.Module):
    """Integrated Multi-Head Self-Attention Bottleneck Block for Autoencoder.
    
    Features:
        - 2D Learnable Spatial Positional Embeddings for 28x28 token grid.
        - Pre-LayerNorm Multi-Head Self-Attention with residual connection.
        - Pre-LayerNorm Feed-Forward MLP with GELU and residual connection.
        - Provides attention affinity S and attenuation factor w(i, j) for InfoNCE.
        - Directly trained by Autoencoder reconstruction and adversarial loss.
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        gamma: float = 2.0,
        grid_size: int = 28,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        # 2D Spatial Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, in_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # MHSA with Pre-LayerNorm
        self.norm1 = nn.LayerNorm(in_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-Forward MLP with Pre-LayerNorm
        self.norm2 = nn.LayerNorm(in_dim)
        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, in_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through MHSA Bottleneck.
        
        Args:
            x: Input feature map [B, C, H, W] (typically [B, 256, 28, 28])
            
        Returns:
            out_features: Transformed feature map [B, C, H, W]
            w_attenuation: Penalty attenuation factor w(i, j) [B, N, N]
            attn_weights: Attention matrix S [B, N, N]
        """
        b, c, h, w = x.shape
        n = h * w

        # Spatial Tokenization: [B, C, H, W] -> [B, N, C]
        tokens = x.flatten(2).transpose(1, 2)

        # Add positional embedding if grid matches
        if n == self.num_tokens:
            tokens_pos = tokens + self.pos_embed
        else:
            tokens_pos = tokens

        # MHSA with Residual
        norm_tokens = self.norm1(tokens_pos)
        attn_out, attn_weights = self.mha(norm_tokens, norm_tokens, norm_tokens, need_weights=True)
        tokens = tokens + attn_out

        # MLP with Residual
        tokens = tokens + self.mlp(self.norm2(tokens))

        # Compute attenuation weights w(i, j) using normalized Query/Key tokens
        w_attenuation, s_matrix = compute_attention_attenuation_weights(
            q=norm_tokens,
            k=norm_tokens,
            attn_weights=attn_weights,
            gamma=self.gamma,
        )

        # Reshape back to [B, C, H, W]
        out_features = tokens.transpose(1, 2).contiguous().view(b, c, h, w)

        return out_features, w_attenuation, s_matrix


class ParallelMHSAGuidance(nn.Module):
    """Parallel MHSA Guidance Stream (Decoupled Stream 2).
    
    Computes global attention affinity matrix S and attenuation factor w(i, j)
    for InfoNCE loss without connecting into the main reconstruction pathway.
    Includes a self-reconstruction projection head to ensure learnable gradients flow.
    """

    def __init__(self, in_dim: int = 256, num_heads: int = 4, gamma: float = 2.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.norm = nn.LayerNorm(in_dim)
        self.mha = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, x_e4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x_e4: Feature map from E_4 [B, C, H, W]
            
        Returns:
            w_attenuation: Attenuation factor [B, N, N]
            s_matrix: Affinity matrix S [B, N, N]
            reconstructed_feat: Reconstructed feature [B, C, H, W] for self-training
        """
        b, c, h, w = x_e4.shape
        x_flat = x_e4.detach().flatten(2).transpose(1, 2)  # [B, N, C] - completely decoupled from reconstruction stream
        x_norm = self.norm(x_flat)

        attn_out, attn_weights = self.mha(x_norm, x_norm, x_norm, need_weights=True)
        recon_flat = self.proj(attn_out)
        recon_feat = recon_flat.transpose(1, 2).contiguous().view(b, c, h, w)

        w_attenuation, s_matrix = compute_attention_attenuation_weights(
            q=attn_out,
            k=attn_out,
            attn_weights=attn_weights,
            gamma=self.gamma,
        )

        return w_attenuation, s_matrix, recon_feat


# ==============================================================================
# 3. Encoder (E)
# ==============================================================================

class Encoder(nn.Module):
    """PatchCL-AE Encoder (Paper Table 1, Page 5).
    
    Layers:
        E_1: 224x224, conv 7x7, 64,  InstanceNorm, ReLU
        E_2: 112x112, conv 3x3, 128, InstanceNorm, ReLU, AvgPool
        E_3: 56x56,   conv 3x3, 256, InstanceNorm, ReLU, AvgPool
        E_4: 28x28,   conv 3x3, 256, InstanceNorm, ReLU, AvgPool
        E_5: 28x28,   6 plain conv blocks with 256 channels (Lossy Bottleneck)
    """

    def __init__(self, in_channels: int = 3, num_conv_bottlenecks: int = 6) -> None:
        super().__init__()
        # E_1: conv 7x7, 64, InstanceNorm, ReLU
        self.E_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        # E_2: conv 3x3, 128, InstanceNorm, ReLU, Downsample
        self.E_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # E_3: conv 3x3, 256, InstanceNorm, ReLU, Downsample
        self.E_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # E_4: conv 3x3, 256, InstanceNorm, ReLU, Downsample
        self.E_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # E_5: 6 conv blocks with 256 channels (Lossy Bottleneck)
        self.E_5 = nn.Sequential(*[
            BottleneckBlock(channels=256, use_residual=False)
            for _ in range(num_conv_bottlenecks)
        ])

        self.out_channels = {
            "E_1": 64,
            "E_2": 128,
            "E_3": 256,
            "E_4": 256,
            "E_5": 256,
        }

    def forward_to_e4(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass from E_1 up to E_4."""
        f1 = self.E_1(x)
        f2 = self.E_2(f1)
        f3 = self.E_3(f2)
        f4 = self.E_4(f3)
        return f1, f2, f3, f4

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through encoder E_1 -> E_5."""
        f1, f2, f3, f4 = self.forward_to_e4(x)
        f5 = self.E_5(f4)

        if return_features:
            features = {
                "E_1": f1,
                "E_2": f2,
                "E_3": f3,
                "E_4": f4,
                "E_5": f5,
            }
            return f5, features
        return f5


# ==============================================================================
# 4. Decoder (De)
# ==============================================================================

class Decoder(nn.Module):
    """PatchCL-AE Decoder (Paper Table 1, Page 5).
    
    Layers:
        De_4: 56x56,   upsample, conv 3x3, 256, InstanceNorm, ReLU
        De_3: 112x112, upsample, conv 3x3, 128, InstanceNorm, ReLU
        De_2: 224x224, upsample, conv 3x3, 64,  InstanceNorm, ReLU
        De_1: 224x224, conv 7x7, 3, Tanh
    """

    def __init__(self, out_channels: int = 3) -> None:
        super().__init__()
        # De_4: upsample, conv 3x3, 256, InstanceNorm, ReLU
        self.De_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        )

        # De_3: upsample, conv 3x3, 128, InstanceNorm, ReLU
        self.De_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        # De_2: upsample, conv 3x3, 64, InstanceNorm, ReLU
        self.De_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        # De_1: conv 7x7, 3, Tanh
        self.De_1 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.De_4(x)
        x = self.De_3(x)
        x = self.De_2(x)
        x = self.De_1(x)
        return x


# ==============================================================================
# 5. Discriminator (D)
# ==============================================================================

class Discriminator(nn.Module):
    """PatchCL-AE Discriminator (Paper Table 2, Page 6).
    
    Layers:
        D_1: 112x112, conv 3x3, 64,  InstanceNorm, LeakyReLU, Downsample
        D_2: 56x56,   conv 3x3, 128, InstanceNorm, LeakyReLU, Downsample
        D_3: 28x28,   conv 3x3, 256, InstanceNorm, LeakyReLU, Downsample
        D_4: 14x14,   conv 3x3, 512, InstanceNorm, LeakyReLU, Downsample
        D_5: 7x7,     conv 3x3, 512, InstanceNorm, LeakyReLU, Downsample
        D_6: 4x4,     conv 3x3, 512, InstanceNorm, LeakyReLU, Downsample
        D_7: 8192,    flatten
        D_8: 256,     256d-FC
        D_9: 1,       1d-FC
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.D_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.D_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.D_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.D_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.D_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.D_6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.D_8 = nn.Linear(8192, 256, bias=True)
        self.D_9 = nn.Linear(256, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.D_1(x)
        x = self.D_2(x)
        x = self.D_3(x)
        x = self.D_4(x)
        x = self.D_5(x)
        x = self.D_6(x)
        x = torch.flatten(x, start_dim=1)  # D_7: 8192
        x = self.D_8(x)                    # D_8: 256
        x = self.D_9(x)                    # D_9: 1 (logit)
        return x


# ==============================================================================
# 6. Projection Head (P)
# ==============================================================================

class ProjectionHead(nn.Module):
    """PatchCL-AE Projection Head (Paper Table 3, Page 6).
    
    Structure:
        P_1: in_features -> 256, ReLU
        P_2: 256 -> 256
    """

    def __init__(self, in_features: int = 256, hidden_dim: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class MultiLayerProjection(nn.Module):
    """Collection of projection heads for multi-layer encoder features."""

    def __init__(
        self,
        layer_channels: Dict[str, int],
        hidden_dim: int = 256,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({
            layer_name: ProjectionHead(in_dim, hidden_dim, out_dim)
            for layer_name, in_dim in layer_channels.items()
        })

    def forward(self, feature_map: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Projects a tensor along channel dimension."""
        proj = self.heads[layer_name]
        return proj(feature_map)


# ==============================================================================
# 7. Complete Integrated MHSA PatchCL-AE Architecture
# ==============================================================================

class PatchCL_AE_MHSA_Model(nn.Module):
    """PatchCL-AE Architecture with Integrated Multi-Head Self-Attention.
    
    Integration Modes:
        1. 'bottleneck' (Default & Recommended):
           E_1 -> E_4 -> MHSABottleneckBlock -> E_5 (Conv Bottleneck) -> Decoder -> X_hat.
           - MHSA is directly on the reconstruction path, learning global layer context.
           - The attention matrix S from this trained MHSA modulates InfoNCE.
           - All MHSA parameters receive direct gradient flow from reconstruction & GAN loss.
           - X_hat is directly enhanced with global contextual consistency.
        2. 'parallel':
           Decoupled dual-stream where MHSA branches parallel from E_4 and modulates InfoNCE.
        3. 'both':
           Bottleneck MHSA on reconstruction path + Parallel guidance MHSA.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_layers: Optional[List[str]] = None,
        proj_dim: int = 256,
        num_heads: int = 4,
        gamma: float = 2.0,
        mhsa_mode: str = "parallel",
        num_conv_bottlenecks: int = 6,
    ) -> None:
        super().__init__()
        self.mhsa_mode = mhsa_mode
        self.gamma = gamma

        # 1. Base Encoder and Decoder
        self.encoder = Encoder(in_channels=in_channels, num_conv_bottlenecks=num_conv_bottlenecks)
        self.decoder = Decoder(out_channels=in_channels)

        # 2. Multi-scale feature layers for contrastive learning
        if feature_layers is None:
            feature_layers = ["E_1", "E_2", "E_3", "E_4"]
        self.feature_layers = feature_layers

        proj_channels = {l: self.encoder.out_channels[l] for l in self.feature_layers}
        self.projection = MultiLayerProjection(
            layer_channels=proj_channels,
            hidden_dim=proj_dim,
            out_dim=proj_dim,
        )

        # 3. MHSA Modules
        if self.mhsa_mode in ("bottleneck", "both"):
            self.mhsa_bottleneck = MHSABottleneckBlock(
                in_dim=self.encoder.out_channels["E_4"],
                num_heads=num_heads,
                mlp_ratio=2.0,
                gamma=gamma,
                grid_size=28,
            )
        else:
            self.mhsa_bottleneck = None

        if self.mhsa_mode in ("parallel", "both"):
            self.parallel_mhsa = ParallelMHSAGuidance(
                in_dim=self.encoder.out_channels["E_4"],
                num_heads=num_heads,
                gamma=gamma,
            )
        else:
            self.parallel_mhsa = None

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]]:
        """Forward pass through PatchCL-AE with MHSA.
        
        Args:
            x: Input image [B, C, H, W]
            return_features: If True, returns (x_recon, feature_dict, w_attenuation)
        """
        # 1. Forward through Encoder E_1 -> E_4
        f1, f2, f3, f4 = self.encoder.forward_to_e4(x)

        w_attenuation = None

        # 2. Process Bottleneck
        if self.mhsa_bottleneck is not None:
            # MHSA Bottleneck on reconstruction path
            z_mhsa, w_bottleneck, _ = self.mhsa_bottleneck(f4)
            z_e5 = self.encoder.E_5(z_mhsa)
            w_attenuation = w_bottleneck
        else:
            z_e5 = self.encoder.E_5(f4)

        # 3. Parallel MHSA Guidance (if enabled)
        recon_feat_mhsa = None
        if self.parallel_mhsa is not None and (self.training or return_features):
            w_par, _, recon_feat_mhsa = self.parallel_mhsa(f4)
            if w_attenuation is None:
                w_attenuation = w_par

        # 4. Forward through Decoder
        x_rec = self.decoder(z_e5)

        features = {
            "E_1": f1,
            "E_2": f2,
            "E_3": f3,
            "E_4": f4,
            "E_5": z_e5,
        }
        if recon_feat_mhsa is not None:
            features["recon_feat_mhsa"] = recon_feat_mhsa

        if return_features:
            return x_rec, features, w_attenuation
        return x_rec

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode image and return bottleneck + layer features."""
        f1, f2, f3, f4 = self.encoder.forward_to_e4(x)
        if self.mhsa_bottleneck is not None:
            z_mhsa, _, _ = self.mhsa_bottleneck(f4)
            z_e5 = self.encoder.E_5(z_mhsa)
        else:
            z_e5 = self.encoder.E_5(f4)
        features = {
            "E_1": f1,
            "E_2": f2,
            "E_3": f3,
            "E_4": f4,
            "E_5": z_e5,
        }
        return z_e5, features

    def decode(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck into reconstructed image."""
        return self.decoder(bottleneck)

    def project_feature(self, feat: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Projects a feature map or sampled vectors through layer's projection head."""
        return self.projection(feat, layer_name)


# Aliases for convenience and backward compatibility
DecoupledPatchCLAE = PatchCL_AE_MHSA_Model
PatchCL_AE = PatchCL_AE_MHSA_Model


def create_patchcl_ae_models(
    in_channels: int = 3,
    feature_layers: Optional[List[str]] = None,
    num_heads: int = 4,
    gamma: float = 2.0,
    proj_dim: int = 256,
    mhsa_mode: str = "parallel",
    num_conv_bottlenecks: int = 6,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Tuple[PatchCL_AE_MHSA_Model, Discriminator]:
    """Factory function to initialize MHSA-enhanced PatchCL-AE generator and Discriminator."""
    generator = PatchCL_AE_MHSA_Model(
        in_channels=in_channels,
        feature_layers=feature_layers,
        proj_dim=proj_dim,
        num_heads=num_heads,
        gamma=gamma,
        mhsa_mode=mhsa_mode,
        num_conv_bottlenecks=num_conv_bottlenecks,
    )
    discriminator = Discriminator(in_channels=in_channels)
    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)
    return generator, discriminator
