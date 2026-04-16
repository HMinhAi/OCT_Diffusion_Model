# OCT Diffusion Model

Diffusion-based unsupervised anomaly detection for OCT images using inversion, feature-deviation correction, simplex-noise robustness, and attention-guided multi-source anomaly fusion.

## Overview

This project implements an anomaly detection pipeline with the following design:

1. Train a DDPM on normal OCT images only.
2. Perform inversion at inference to get reconstruction-free anomaly cues.
3. Apply deviation correction using a normal feature distribution (ResNet-50 + Mahalanobis distance).
4. Add simplex noise perturbations and aggregate anomaly responses for robustness.
5. Fuse multi-source anomaly maps with a learnable attention module.

Core formulas:

- Corrected score: `S_corr = S_diffusion - lambda * S_feature`
- Final fusion: `A_final = alpha * A_diff + beta * A_feat + gamma * A_noise`

## Main Features

- DDPM with U-Net denoiser trained on normal class only.
- Inversion-based anomaly maps from image-space and optional noise-space errors.
- Feature deviation correction via Mahalanobis distance.
- Simplex noise injection during inference (multi-pass aggregation).
- Attention-guided fusion over three anomaly sources.
- Image-level AUROC and pixel-level AUROC (when masks are available).
- Visualization outputs with anomaly heatmap overlays.
- Weights & Biases logging support.
- Config-driven ablation switches.

## Project Structure

```text
.
├── config.yaml
├── train.py
├── eval.py
├── requirements.txt
├── datasets/
│   └── oct_dataset.py
├── models/
│   ├── diffusion_model.py
│   ├── inversion.py
│   ├── feature_extractor.py
│   └── attention_fusion.py
└── utils/
    ├── metrics.py
    └── visualization.py
```

## Installation

### 1. Create environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install GPU PyTorch wheel (recommended)

Example for CUDA 12.4:

```bash
pip install --upgrade --force-reinstall --no-cache-dir \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
```

Verify CUDA availability:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Dataset Setup

Expected Kermany-style layout:

```text
datasets/OCT2017/
├── train/
│   ├── NORMAL/
│   ├── CNV/
│   ├── DME/
│   └── DRUSEN/
├── test/
│   ├── NORMAL/
│   ├── CNV/
│   ├── DME/
│   └── DRUSEN/
└── masks/                       # optional for pixel-level AUROC
    └── ...
```

Notes:

- Training uses only `train/NORMAL` by design.
- Test uses normal + abnormal classes.
- Pixel-level AUROC is computed only when masks are found.

Optional Kaggle auto-download:

1. Install Kaggle CLI and configure API token.
2. Set in `config.yaml`:
   - `dataset.use_kaggle_download: true`
   - `dataset.kaggle.dataset_slug: paultimothymooney/kermany2018`

## Configuration

Main runtime configuration lives in `config.yaml`:

- `project`: seed, output/checkpoint dirs, workers, device.
- `dataset`: paths, class names, transform normalization, optional Kaggle download.
- `train`: batch size, epochs, LR, AMP, pseudo-label quantile.
- `diffusion`: DDPM schedule and inversion steps.
- `inference`: simplex settings, correction lambda.
- `feature`: ResNet layer and Mahalanobis options.
- `fusion`: attention module hidden channels.
- `ablation`: enable/disable each branch.
- `wandb`: logging settings.
- `eval`: eval batch size and visualization count.

## Training

Run full pipeline training:

```bash
python train.py --config config.yaml
```

Disable wandb:

```bash
python train.py --config config.yaml --disable-wandb
```

What `train.py` does:

1. Train DDPM.
2. Fit feature distribution on normal train set.
3. Train attention fusion with pseudo labels.
4. Run evaluation and print final AUROC.

## Evaluation

Run standalone evaluation from saved checkpoints:

```bash
python eval.py --config config.yaml --checkpoint-dir checkpoints --output-dir outputs
```

Disable wandb:

```bash
python eval.py --config config.yaml --disable-wandb
```

Expected outputs:

- `Image-level AUROC` always.
- `Pixel-level AUROC` if masks exist, otherwise `nan` with warning.

## Ablation Study

Toggle branches in `config.yaml` under `ablation`:

- `enable_diffusion`
- `enable_feature_correction`
- `enable_simplex_noise`
- `enable_attention_fusion`
- `enable_noise_space_error`

This allows controlled experiments on each component.

## Outputs

By default:

- Checkpoints: `checkpoints/`
  - `diffusion_best.pt`, `diffusion_last.pt`, `feature_stats.pt`, `fusion_best.pt`
- Visualizations: `outputs/eval_visuals/`
  - Input image + diffusion map + feature map + noise map + fused map

## Common Issues

1. `CUDA requested but unavailable`:
   - Check NVIDIA driver, CUDA-compatible torch wheel, and `torch.cuda.is_available()`.
2. `No diffusion checkpoint found` in eval:
   - Run training first or provide `--checkpoint-dir` correctly.
3. `Pixel AUROC is skipped`:
   - Add mask files under `dataset.mask_dir` structure.
4. Kaggle download does not start:
   - Install Kaggle CLI and configure credentials.

## Reproducibility

- Global random seed is configured in `project.seed`.
- Device selection is automatic (`cuda` when available, else `cpu`).

## Acknowledgment

- Dataset style follows Kermany OCT splits.
- PyTorch ecosystem: torch, torchvision, sklearn, matplotlib, wandb.