"""PatchCL-AE with Integrated Multi-Head Self-Attention (MHSA).

Reference:
    Shuai Lu, Weihang Zhang, Jia Guo, Hanruo Liu, Huiqi Li, Ningli Wang.
    "PatchCL-AE: Anomaly detection for medical images using patch-wise contrastive learning-based auto-encoder"
    Computerized Medical Imaging and Graphics 114 (2024) 102366.
"""

from .anomaly_score import (
    anomaly_map_to_score,
    apply_global_thresholding,
    apply_quantile_soft_thresholding,
    compute_anomaly_score_simple,
    compute_global_baseline_thresholds,
    compute_layer_similarity_map,
    compute_patchcl_anomaly_map,
    load_global_thresholds,
    save_global_thresholds,
)
from .dataset import (
    MedicalAnomalyDataset,
    create_patchcl_dataloaders,
    get_patchcl_transforms,
)
from .losses import (
    AttentionWeightedPatchContrastiveLoss,
    EdgeLoss,
    ImageAdversarialLoss,
    PatchCL_TotalLoss,
    PatchContrastiveLoss,
    SobelEdgeLoss,
    compute_attention_guided_infonce_loss,
)
from .metrics import calculate_anomaly_metrics
from .models import (
    BottleneckBlock,
    ConvBlock,
    Decoder,
    DecoupledPatchCLAE,
    Discriminator,
    Encoder,
    MHSABottleneckBlock,
    MultiLayerProjection,
    ParallelMHSAGuidance,
    PatchCL_AE,
    PatchCL_AE_MHSA_Model,
    ProjectionHead,
    ResnetBlock,
    create_patchcl_ae_models,
)
from .trainer import PatchCLTrainer

__all__ = [
    "ConvBlock",
    "BottleneckBlock",
    "ResnetBlock",
    "MHSABottleneckBlock",
    "ParallelMHSAGuidance",
    "Encoder",
    "Decoder",
    "Discriminator",
    "ProjectionHead",
    "MultiLayerProjection",
    "DecoupledPatchCLAE",
    "PatchCL_AE_MHSA_Model",
    "PatchCL_AE",
    "create_patchcl_ae_models",
    "ImageAdversarialLoss",
    "AttentionWeightedPatchContrastiveLoss",
    "PatchContrastiveLoss",
    "SobelEdgeLoss",
    "EdgeLoss",
    "PatchCL_TotalLoss",
    "compute_attention_guided_infonce_loss",
    "compute_layer_similarity_map",
    "compute_anomaly_score_simple",
    "apply_global_thresholding",
    "apply_quantile_soft_thresholding",
    "compute_global_baseline_thresholds",
    "compute_patchcl_anomaly_map",
    "save_global_thresholds",
    "load_global_thresholds",
    "anomaly_map_to_score",
    "calculate_anomaly_metrics",
    "MedicalAnomalyDataset",
    "get_patchcl_transforms",
    "create_patchcl_dataloaders",
    "PatchCLTrainer",
]
