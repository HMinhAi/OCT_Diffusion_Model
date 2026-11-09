"""Loss functions for PatchCL-AE (Lu et al., CMIG 2024) with Attention Guidance.

Paper Reference:
    Shuai Lu, Weihang Zhang, Jia Guo, Hanruo Liu, Huiqi Li, Ningli Wang.
    "PatchCL-AE: Anomaly detection for medical images using patch-wise
    contrastive learning-based auto-encoder."
    Computerized Medical Imaging and Graphics 114 (2024) 102366.
    DOI: 10.1016/j.compmedimag.2024.102366

Loss Formulations from Paper:
    1. Section 3.1.2: Image-wise Loss (Eq. 1)
       \\mathcal{L}_{Img}(E, De, D) = \\mathbb{E}_X [\\log D(X)] + \\mathbb{E}_X [\\log(1 - D(\\hat{X}))]
       where \\hat{X} = De(E(X + \\text{Noise})), \\text{Noise} \\sim \\mathcal{N}(0, \\sigma^2 I) with \\sigma = 0.05.
       - Discriminator Loss (Algorithm 1, Step 6):
         \\mathcal{L}_D = -0.5 * \\mathbb{E}_X [\\log D(X)] - 0.5 * \\mathbb{E}_X [\\log(1 - D(\\hat{X}))]
       - Generator Adversarial Loss (Algorithm 1, Step 7):
         \\mathcal{L}_{adv} = -\\mathbb{E}_X [\\log D(\\hat{X})]

    2. Section 3.1.3: Patch-wise Contrastive Learning Loss (Eq. 2, Eq. 3)
       Across L layers (E_1 to E_5), for K = 256 sampled locations:
       Query:     \\hat{z}_s^l = P_l(E_l(\\hat{x}_s))  (from reconstructed image \\hat{X})
       Positive:  z_s^{+, l} = P_l(E_l(x_s^+))         (from raw input image X at location s)
       Negative:  z_{s'}^{-, l} = P_l(E_l(x_{s'}^-))     (from raw input image X at locations s' != s)
       All features are L2-normalized: sim(u, v) = u^T v / (||u||_2 ||v||_2).
       \\mathcal{L}_{Patch} = \\sum_{l=1}^L \\sum_{s=1}^{S_l} \\ell(\\hat{z}_s^l, z_s^{+, l}, z_{s'}^{-, l})
       \\ell(\\hat{z}_s^l, z_s^{+, l}, z_{s'}^{-, l}) = - \\log \\left[ \\frac{\\exp(\\text{sim}(\\hat{z}_s^l, z_s^{+, l}) / \\tau)}{\\exp(\\text{sim}(\\hat{z}_s^l, z_s^{+, l}) / \\tau) + \\sum_{s' \\neq s} \\exp(\\text{sim}(\\hat{z}_s^l, z_{s'}^{-, l}) / \\tau)} \\right]

    3. Section 3.1.4: Total Loss (Eq. 4)
       \\mathcal{L}_{Total} = \\mathcal{L}_{Img}(E, De, D) + \\mathcal{L}_{Patch}(E, De)
       Generator Objective:
       \\mathcal{L}_{Total} = \\lambda_{adv} \\mathcal{L}_{adv} + \\lambda_{patch} \\mathcal{L}_{Patch}
       Standard paper ratio: \\lambda_{adv} = 1.0, \\lambda_{patch} = 1.0 (Eq. 4).
       Note: The paper strictly uses NO L1 pixel loss and NO Sobel edge loss.

Attention Matrix Application (Decoupled Guidance Innovation):
    The parallel MHSA stream on E_4(X) computes the spatial correlation matrix S in [0, 1]^{N x N}.
    For mutually similar/symmetric normal regions (e.g. bilateral retinal layers or symmetric lungs),
    S_{ij} -> 1.
    The attenuation factor w(i, j) = (1 - S_{ij})^\\gamma (with \\gamma = 2.0, detached) modulates
    the negative penalty terms in the InfoNCE denominator:
       \\text{Denominator} = \\exp(\\text{sim}(\\hat{z}_i^l, z_i^{+, l}) / \\tau) + \\sum_{j \\neq i} w(i, j) \\cdot \\exp(\\text{sim}(\\hat{z}_i^l, z_j^{-, l}) / \\tau)
    - If S_{ij} -> 1: w(i, j) -> 0, eliminating bilateral false-negative semantic collision.
    - If S_{ij} -> 0: w(i, j) -> 1, preserving full contrastive repulsion.
    - If w(i, j) is None: strictly degenerates to standard paper InfoNCE (Eq. 3).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Global Image-Wise Adversarial Loss (Paper Section 3.1.2, Eq. 1 & Alg. 1)
# ==============================================================================
class ImageAdversarialLoss(nn.Module):
    """Global image-wise adversarial loss (Eq. 1, Page 3 & Algorithm 1).

    L_Img(E, De, D) = E_X [log D(X)] + E_X [log(1 - D(X_hat))]
    where X_hat = De(E(X + Noise)).
    """

    def __init__(self, label_smoothing: float = 1.0) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def discriminator_loss(
        self, d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes loss for Discriminator D (Paper Algorithm 1, Step 6).

        Args:
            d_real_logits: Logits from D(X)
            d_fake_logits: Logits from D(X_hat.detach())

        Returns:
            loss_d: Total discriminator loss (0.5 * loss_real + 0.5 * loss_fake)
            loss_real: Discriminator loss on real images
            loss_fake: Discriminator loss on reconstructed images
        """
        if self.label_smoothing < 1.0:
            real_targets = torch.full_like(d_real_logits, self.label_smoothing)
        else:
            real_targets = torch.ones_like(d_real_logits)
        fake_targets = torch.zeros_like(d_fake_logits)

        loss_real = self.criterion(d_real_logits, real_targets)
        loss_fake = self.criterion(d_fake_logits, fake_targets)
        loss_d = 0.5 * (loss_real + loss_fake)

        return loss_d, loss_real, loss_fake

    def generator_loss(self, d_fake_logits: torch.Tensor) -> torch.Tensor:
        """Computes adversarial loss for Generator (Encoder + Decoder) (Paper Algorithm 1, Step 7).

        Args:
            d_fake_logits: Logits from D(X_hat)

        Returns:
            loss_adv: Adversarial loss for generator BCEWithLogits(D(X_hat), 1.0)
        """
        real_targets = torch.ones_like(d_fake_logits)
        return self.criterion(d_fake_logits, real_targets)


# ==============================================================================
# 2. Patch-Wise Contrastive Loss with Attention Guidance (Section 3.1.3, Eq. 3)
# ==============================================================================
def compute_attention_guided_infonce_loss(
    f_real: torch.Tensor,
    f_rec: torch.Tensor,
    w_attenuation: Optional[torch.Tensor] = None,
    tau: float = 0.07,
) -> torch.Tensor:
    """Attention-guided InfoNCE patch loss following Eq. (3) of paper (Lu et al., 2024).

    Mathematical Formulation:
        Query:      \\hat{z}_i = P_l(E_l(\\hat{x}_i)) from reconstructed image \\hat{X} (f_rec)
        Positive:   z_i^+ = P_l(E_l(x_i^+)) from input raw image X at location i (f_real[:, i])
        Negative:   z_j^- = P_l(E_l(x_j^-)) from input raw image X at locations j != i (f_real[:, j])

        Cosine Similarity (Paper Section 3.1.3 & Section 4.2.1):
            s_{ij} = \\text{sim}(\\hat{z}_i, z_j) / \\tau = (\\hat{z}_i \\cdot z_j) / \\tau

        Attention-Guided Attenuation on Negatives:
            For mutually similar normal regions (S_{ij} -> 1), the penalty attenuation factor
            w(i, j) = (1 - S_{ij})^\\gamma scales the negative exponential terms in the denominator:
            \\ell_i = -\\log \\left[ \\frac{\\exp(s_{ii})}{\\exp(s_{ii}) + \\sum_{j \\neq i} w(i, j) \\cdot \\exp(s_{ij})} \\right]

        When w(i, j) is None (or 1.0):
            \\ell_i = -\\log \\left[ \\frac{\\exp(s_{ii})}{\\sum_{j=1}^N \\exp(s_{ij})} \\right] = \\text{CrossEntropy}(s_i, i)
            which matches the exact baseline formulation of the paper.

    Args:
        f_real: [B, N, C] Feature vectors from original image X (Keys/Anchors)
        f_rec:  [B, N, C] Feature vectors from reconstructed image X_hat (Queries)
        w_attenuation: [B, N, N] Optional attenuation weights w(i, j) = (1 - S_{ij})^gamma
        tau: Temperature parameter (default: 0.07)

    Returns:
        Scalar InfoNCE loss averaged across all sampled patches and batch.
    """
    b, n, c = f_real.shape
    f_real = F.normalize(f_real, p=2, dim=-1)
    f_rec = F.normalize(f_rec, p=2, dim=-1)

    # Cosine similarity matrix between queries (reconstructed) and keys (real):
    # sim_matrix[b, i, j] = sim(z_fake_i, z_real_j) / tau
    sim_matrix = torch.bmm(f_rec, f_real.transpose(1, 2)) / tau  # [B, N, N]

    # Baseline mode: standard Cross-Entropy (Paper Eq. 3)
    if w_attenuation is None:
        targets = torch.arange(n, device=f_real.device).repeat(b, 1)
        return F.cross_entropy(sim_matrix.view(b * n, n), targets.view(b * n), reduction="mean")

    # Attention-guided mode: modulate negative penalties using w(i, j) = (1 - S_ij)^gamma
    w_att = w_attenuation.detach()  # ngắt gradient chống shortcut learning

    # Positive logits along diagonal: s_{ii} = sim(z_fake_i, z_real_i) / tau
    pos_logits = torch.diagonal(sim_matrix, dim1=1, dim2=2)  # [B, N]

    # Numerically stable Log-Sum-Exp by subtracting row maximum
    max_logits = sim_matrix.max(dim=-1, keepdim=True).values.detach()  # [B, N, 1]
    logits_shifted = sim_matrix - max_logits  # [B, N, N]
    pos_shifted = pos_logits - max_logits.squeeze(-1)  # [B, N]

    exp_shifted = torch.exp(logits_shifted)  # [B, N, N]
    mask = torch.eye(n, device=f_real.device, dtype=torch.bool).unsqueeze(0)  # [1, N, N]

    # Scale negative samples (j != i) by attention attenuation weight w(i, j)
    weighted_neg_exp = exp_shifted * w_att
    weighted_neg_exp = weighted_neg_exp.masked_fill(mask, 0.0)  # remove diagonal (positive pair)

    pos_exp = torch.exp(pos_shifted)  # [B, N]
    neg_sum = weighted_neg_exp.sum(dim=-1)  # [B, N]
    denom = pos_exp + neg_sum + 1e-8

    loss = -pos_shifted + torch.log(denom)
    return loss.mean()


class AttentionWeightedPatchContrastiveLoss(nn.Module):
    """Attention-Guided Multi-Layer Patch-wise Contrastive Loss (Eq. 2, Eq. 3, Page 4).

    At each layer l in {E_1, E_2, E_3, E_4, E_5}:
        K = 256 random spatial locations are sampled from both real X and reconstructed X_hat.
        Sampled vectors are mapped into 256-dim space through 2-layer MLP projection head P_l.
        The attention-guided InfoNCE loss is computed across all sampled patches.
        Total patch loss is the sum across all layers following Eq. (3).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_patches: int = 256,
        gamma: float = 2.0,
        reduction: str = "sum",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_patches = num_patches
        self.gamma = gamma
        self.reduction = reduction

    def sample_locations(
        self,
        feature_map_real: torch.Tensor,
        feature_map_fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample spatial locations from real and fake feature maps (Paper Section 4.2.1).

        Args:
            feature_map_real: [B, C, H, W] from raw image X
            feature_map_fake: [B, C, H, W] from reconstructed image X_hat

        Returns:
            sampled_real: [B, k, C]
            sampled_fake: [B, k, C]
            perm_idx: Spatial indices tensor [k]
        """
        b, c, h, w = feature_map_real.shape
        num_spatial = h * w
        k = min(self.num_patches, num_spatial)

        flat_real = feature_map_real.view(b, c, num_spatial).permute(0, 2, 1)
        flat_fake = feature_map_fake.view(b, c, num_spatial).permute(0, 2, 1)

        perm_idx = torch.randperm(num_spatial, device=feature_map_real.device)[:k]

        sampled_real = flat_real[:, perm_idx, :]  # [B, k, C]
        sampled_fake = flat_fake[:, perm_idx, :]  # [B, k, C]

        return sampled_real, sampled_fake, perm_idx

    def forward(
        self,
        features_real: Dict[str, torch.Tensor],
        features_fake: Dict[str, torch.Tensor],
        projection_module: nn.Module,
        w_attenuation: Optional[torch.Tensor] = None,
        layers: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute Patch Contrastive InfoNCE loss across layers (Eq. 3, Page 4).

        Args:
            features_real: Features from input X: {layer_name: [B, C, H, W]}
            features_fake: Features from reconstructed X_hat: {layer_name: [B, C, H, W]}
            projection_module: MultiLayerProjection instance
            w_attenuation: Attenuation factor w(i, j) of shape [B, 784, 784] from parallel MHSA
            layers: List of layer names (default: all layers in features_real present in projection)
        """
        if layers is None:
            if hasattr(projection_module, "heads"):
                layers = [k for k in features_real.keys() if k in projection_module.heads]
            else:
                layers = [k for k in features_real.keys() if k in ["E_1", "E_2", "E_3", "E_4", "E_5"]]

        if w_attenuation is not None:
            w_attenuation = w_attenuation.detach()

        device = next(projection_module.parameters()).device
        total_cl_loss = torch.tensor(0.0, device=device)

        for layer_name in layers:
            feat_real = features_real[layer_name]
            feat_fake = features_fake[layer_name]

            # 1. Sample K patches from X and X_hat at the exact same spatial locations
            vec_real, vec_fake, perm_idx = self.sample_locations(feat_real, feat_fake)
            b, k, c = vec_real.shape

            # 2. Project sampled vectors through Projection Head P_l (Paper Section 4.2.1, Table 3)
            z_real = projection_module(vec_real.contiguous().view(-1, c), layer_name).view(b, k, -1)
            z_fake = projection_module(vec_fake.contiguous().view(-1, c), layer_name).view(b, k, -1)

            # 3. Trích xuất ma trận trọng số w_sub cho tập patch được lấy mẫu
            # Ánh xạ chỉ số không gian về lưới 28x28 của attention map
            w_sub = None
            if w_attenuation is not None:
                if layer_name in ["E_4", "E_5"]:
                    # Lưới 28x28 (N=784) khớp trực tiếp
                    w_sub = w_attenuation[:, perm_idx, :][:, :, perm_idx]  # [B, k, k]
                elif layer_name == "E_3":
                    # Lưới 56x56 -> 28x28: (r_56 // 2, c_56 // 2)
                    r_56 = perm_idx // 56
                    c_56 = perm_idx % 56
                    s_mapped = (r_56 // 2) * 28 + (c_56 // 2)
                    w_sub = w_attenuation[:, s_mapped, :][:, :, s_mapped]  # [B, k, k]
                elif layer_name == "E_2":
                    # Lưới 112x112 -> 28x28: (r_112 // 4, c_112 // 4)
                    r_112 = perm_idx // 112
                    c_112 = perm_idx % 112
                    s_mapped = (r_112 // 4) * 28 + (c_112 // 4)
                    w_sub = w_attenuation[:, s_mapped, :][:, :, s_mapped]  # [B, k, k]
                elif layer_name == "E_1":
                    # Lưới 224x224 -> 28x28: (r_224 // 8, c_224 // 8)
                    r_224 = perm_idx // 224
                    c_224 = perm_idx % 224
                    s_mapped = (r_224 // 8) * 28 + (c_224 // 8)
                    w_sub = w_attenuation[:, s_mapped, :][:, :, s_mapped]  # [B, k, k]

            # 4. Tính InfoNCE Loss cho tầng l
            layer_loss = compute_attention_guided_infonce_loss(
                f_real=z_real,
                f_rec=z_fake,
                w_attenuation=w_sub,
                tau=self.temperature,
            )
            total_cl_loss = total_cl_loss + layer_loss

        # Paper Eq. (3) sums across all layers L: \sum_{l=1}^L \sum_{s=1}^{S_l} \ell(...)
        if self.reduction == "mean":
            return total_cl_loss / max(1, len(layers))
        return total_cl_loss


# Alias for backward compatibility
PatchContrastiveLoss = AttentionWeightedPatchContrastiveLoss


# ==============================================================================
# 3. Total Loss for PatchCL-AE (Paper Section 3.1.4, Eq. 4 & Algorithm 1)
# ==============================================================================
class PatchCL_TotalLoss(nn.Module):
    """Standard Total Loss for PatchCL-AE (Paper Eq. 4):

        L_Total = L_Img(E, De, D) + L_Patch(E, De)
                = lambda_adv * L_adv + lambda_patch * L_patch

    Standard Paper Configuration (Lu et al., 2024):
        lambda_adv   = 1.0 (Eq. 4: 1:1 ratio)
        lambda_patch = 1.0 (Eq. 4: 1:1 ratio)
        tau          = 0.07 (Section 4.2.1)
        num_patches  = 256 (Section 4.2.1)

    Note: The paper strictly contains NO L1 pixel loss and NO Sobel edge loss.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_patches: int = 256,
        lambda_patch: float = 1.0,
        lambda_adv: float = 1.0,
        gamma: float = 2.0,
        label_smoothing: float = 1.0,
        reduction: str = "sum",
        lambda_l1: float = 0.0,
        lambda_edge: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.adv_loss = ImageAdversarialLoss(label_smoothing=label_smoothing)
        self.patch_loss = AttentionWeightedPatchContrastiveLoss(
            temperature=temperature,
            num_patches=num_patches,
            gamma=gamma,
            reduction=reduction,
        )
        self.lambda_patch = lambda_patch
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_edge = lambda_edge

        # Auxiliary edge loss kept only for legacy callers if lambda_edge > 0
        self._edge_loss_module = None

    def set_lambda_mem(self, lambda_mem: float) -> None:
        """No-op kept for backward compatibility."""
        pass

    def forward_generator(
        self,
        d_fake_logits: torch.Tensor,
        features_real: Dict[str, torch.Tensor],
        features_fake: Dict[str, torch.Tensor],
        projection_module: nn.Module,
        w_attenuation: Optional[torch.Tensor] = None,
        layers: Optional[List[str]] = None,
        real_images: Optional[torch.Tensor] = None,
        recon_images: Optional[torch.Tensor] = None,
        return_all: bool = False,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Compute standard Generator loss combining Adversarial loss and Patch Contrastive loss (Paper Eq. 4):

            L_Total = lambda_adv * L_adv + lambda_patch * L_patch

        Args:
            d_fake_logits: Logits from D(X_hat)
            features_real: Multi-layer features from raw image X: {layer_name: [B, C, H, W]}
            features_fake: Multi-layer features from reconstructed image X_hat: {layer_name: [B, C, H, W]}
            projection_module: MultiLayerProjection instance
            w_attenuation: Attenuation factor w(i, j) from parallel MHSA
            layers: List of layer names to evaluate contrastive loss on
            real_images: Ground-truth/reference image tensor X [B, C, H, W] (auxiliary)
            recon_images: Reconstructed image tensor X_hat [B, C, H, W] (auxiliary)
            return_all: If True, returns (loss_g_total, loss_adv, loss_patch, loss_l1, loss_edge).
                        If False (default), returns (loss_g_total, loss_adv, loss_patch).
            return_dict: If True, returns dictionary of loss components.

        Returns:
            loss_g_total, loss_adv, loss_patch [, loss_l1, loss_edge]
        """
        # 1. Global Image Adversarial Loss (Paper Section 3.1.2, Eq. 1)
        loss_adv = self.adv_loss.generator_loss(d_fake_logits)

        # 2. Local Patch Contrastive InfoNCE Loss with Attention Guidance (Paper Section 3.1.3, Eq. 3)
        loss_patch = self.patch_loss(
            features_real=features_real,
            features_fake=features_fake,
            projection_module=projection_module,
            w_attenuation=w_attenuation,
            layers=layers,
        )

        # 3. Standard Paper Total Loss (Paper Section 3.1.4, Eq. 4)
        loss_g_total = self.lambda_adv * loss_adv + self.lambda_patch * loss_patch

        # Auxiliary pixel/edge loss support (only active if user explicitly specifies lambda_l1 > 0 or lambda_edge > 0)
        device = d_fake_logits.device
        loss_l1 = torch.tensor(0.0, device=device)
        loss_edge = torch.tensor(0.0, device=device)

        if (self.lambda_l1 > 0 or self.lambda_edge > 0 or return_all) and real_images is not None and recon_images is not None:
            if self.lambda_l1 > 0 or return_all:
                loss_l1 = F.l1_loss(recon_images, real_images)
                loss_g_total = loss_g_total + self.lambda_l1 * loss_l1
            if self.lambda_edge > 0 or return_all:
                if self._edge_loss_module is None:
                    self._edge_loss_module = SobelEdgeLoss().to(device)
                loss_edge = self._edge_loss_module(recon_images, real_images)
                loss_g_total = loss_g_total + self.lambda_edge * loss_edge

        if return_dict:
            res = {
                "loss_total": loss_g_total,
                "loss_adv": loss_adv,
                "loss_patch": loss_patch,
            }
            if return_all:
                res["loss_l1"] = loss_l1
                res["loss_edge"] = loss_edge
            return res

        if return_all:
            return loss_g_total, loss_adv, loss_patch, loss_l1, loss_edge

        return loss_g_total, loss_adv, loss_patch

    def forward_discriminator(
        self, d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute discriminator loss (Paper Algorithm 1, Step 6).

        Args:
            d_real_logits: Logits from D(X)
            d_fake_logits: Logits from D(X_hat.detach())

        Returns:
            loss_d, loss_real, loss_fake
        """
        return self.adv_loss.discriminator_loss(d_real_logits, d_fake_logits)


# ==============================================================================
# Auxiliary Modules (Retained for Backward Compatibility)
# ==============================================================================
class SobelEdgeLoss(nn.Module):
    """Auxiliary Sobel Edge Loss (Non-paper component, retained for backward compatibility)."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", sobel_x)
        self.register_buffer("kernel_y", sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        kx = self.kernel_x.repeat(c, 1, 1, 1).to(dtype=pred.dtype, device=pred.device)
        ky = self.kernel_y.repeat(c, 1, 1, 1).to(dtype=pred.dtype, device=pred.device)

        grad_x_pred = F.conv2d(pred, kx, padding=1, groups=c)
        grad_y_pred = F.conv2d(pred, ky, padding=1, groups=c)
        grad_x_target = F.conv2d(target, kx, padding=1, groups=c)
        grad_y_target = F.conv2d(target, ky, padding=1, groups=c)

        loss_x = F.l1_loss(grad_x_pred, grad_x_target)
        loss_y = F.l1_loss(grad_y_pred, grad_y_target)
        return 0.5 * (loss_x + loss_y)


EdgeLoss = SobelEdgeLoss
