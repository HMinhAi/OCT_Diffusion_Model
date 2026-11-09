"""Dataset loader for PatchCL-AE medical anomaly detection.

Reference:
    Section 4.1: Datasets (Retinal OCT, Chest X-ray, Brain MRI)
    Section 4.2.1: Implementation details:
        "The model f is trained on the training set D_train containing only normal samples.
         During testing, the model will be evaluated on D_test to differentiate normal and anomaly samples.
         The scale of the original image is resized to 256x256 (or 224x224)."
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MedicalAnomalyDataset(Dataset):
    """General Medical Anomaly Detection Dataset."""

    def __init__(
        self,
        subset_dataset: Union[Dataset, Subset],
        is_train: bool,
        normal_class_idx: int,
    ) -> None:
        self.dataset = subset_dataset
        self.is_train = is_train
        self.normal_class_idx = normal_class_idx

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        if isinstance(self.dataset, Subset):
            original_idx = self.dataset.indices[index]
            image, original_label = self.dataset.dataset[original_idx]
            img_path, _ = self.dataset.dataset.samples[original_idx]
        else:
            image, original_label = self.dataset[index]
            img_path, _ = self.dataset.samples[index]

        # Normal is 0, Abnormal is 1
        label = 0 if original_label == self.normal_class_idx else 1

        return {
            "image": image,
            "label": label,
            "path": img_path,
        }

    def __len__(self) -> int:
        return len(self.dataset)


def get_patchcl_transforms(
    img_size: int = 224,
    resize_size: int = 256,
    is_train: bool = True,
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> transforms.Compose:
    """Standard transform pipeline mapping images into [-1, 1] range for Tanh decoder.
    
    Paper Section 4.2.1:
        "The scale of the original image is resized to 256x256."
    Table 1:
        Network input tensor is 224x224.
    Pipeline:
        Resize to 256x256 -> RandomCrop(224) (Train) / CenterCrop(224) (Test).
    """
    ops = [transforms.Resize((resize_size, resize_size))]
    if resize_size != img_size:
        ops.append(transforms.RandomCrop(img_size) if is_train else transforms.CenterCrop(img_size))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
    ])
    return transforms.Compose(ops)


def create_patchcl_dataloaders(
    root_path: str,
    train_dir: str = "train",
    test_dir: str = "test",
    normal_class_name: str = "NORMAL",
    img_size: int = 224,
    resize_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 2,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Create train (normal-only) and test (normal + anomalies) DataLoaders.
    
    Args:
        root_path: Dataset root directory (e.g. datasets/OCT2017)
        train_dir: Subdirectory name for train set
        test_dir: Subdirectory name for test set
        normal_class_name: Name of folder containing normal samples
        img_size: Target cropped dimension for network (default: 224 per Table 1)
        resize_size: Initial resized dimension (default: 256 per Section 4.2.1)
        batch_size: Minibatch size (default: 16)
        num_workers: DataLoader worker count
        max_train_samples: Optional limit for train images
        max_test_samples: Optional limit for test images
    """
    train_transform = get_patchcl_transforms(img_size=img_size, resize_size=resize_size, is_train=True)
    test_transform = get_patchcl_transforms(img_size=img_size, resize_size=resize_size, is_train=False)

    # 0. Auto-resolve root_path if train_dir is not found directly
    train_full_path = os.path.join(root_path, train_dir)
    if not os.path.exists(train_full_path):
        resolved_root = None
        # Check if root_path exists and look for train_dir inside its subdirectories
        if os.path.exists(root_path):
            for r, dirs, _ in os.walk(root_path):
                if train_dir in dirs:
                    resolved_root = r
                    break
        # Also check common Colab parent/sub patterns if not found
        if resolved_root is None:
            common_candidates = [
                "/content/data/dataset/chest_xray",
                "/content/data/chest_xray",
                "/content/data/dataset",
                "/content/data/OCT2017",
                "/content/data",
                "datasets/chest_xray",
                "datasets/OCT2017",
            ]
            for cand in common_candidates:
                if os.path.exists(cand):
                    for r, dirs, _ in os.walk(cand):
                        if train_dir in dirs:
                            resolved_root = r
                            break
                    if resolved_root:
                        break
        if resolved_root:
            print(f"[DataLoader] Auto-detected actual dataset root at: '{resolved_root}' (original was: '{root_path}')")
            root_path = resolved_root
            train_full_path = os.path.join(root_path, train_dir)

    # 1. Training set: ONLY normal images
    train_folder = datasets.ImageFolder(root=train_full_path, transform=train_transform)
    train_normal_idx = train_folder.class_to_idx.get(normal_class_name)
    if train_normal_idx is None:
        for c_name, c_idx in train_folder.class_to_idx.items():
            if c_name.lower() == normal_class_name.lower():
                train_normal_idx = c_idx
                break
    if train_normal_idx is None:
        raise ValueError(
            f"Class '{normal_class_name}' not found in train dir. Available: {train_folder.classes}"
        )

    # Filter only normal samples
    train_indices = [
        idx for idx, (_, lbl) in enumerate(train_folder.imgs) if lbl == train_normal_idx
    ]
    if max_train_samples is not None:
        train_indices = train_indices[:int(max_train_samples)]

    train_dataset = MedicalAnomalyDataset(
        Subset(train_folder, train_indices), is_train=True, normal_class_idx=train_normal_idx
    )

    # 2. Test set: both normal and abnormal
    test_full_path = os.path.join(root_path, test_dir)
    if not os.path.exists(test_full_path):
        if os.path.exists(os.path.join(root_path, "val")):
            test_full_path = os.path.join(root_path, "val")
            print(f"[DataLoader] Using '{test_full_path}' as test set.")
    test_folder = datasets.ImageFolder(root=test_full_path, transform=test_transform)
    test_normal_idx = test_folder.class_to_idx.get(normal_class_name)
    if test_normal_idx is None:
        for c_name, c_idx in test_folder.class_to_idx.items():
            if c_name.lower() == normal_class_name.lower():
                test_normal_idx = c_idx
                break
    if test_normal_idx is None:
        raise ValueError(
            f"Class '{normal_class_name}' not found in test dir. Available: {test_folder.classes}"
        )

    all_classes = test_folder.classes
    num_classes = len(all_classes)

    if max_test_samples is not None:
        # Sample evenly across classes
        samples_per_class = int(max_test_samples) // num_classes
        indices_per_class = {c: [] for c in range(num_classes)}
        for idx, (_, lbl) in enumerate(test_folder.imgs):
            indices_per_class[lbl].append(idx)

        test_indices = []
        for lbl, idxs in indices_per_class.items():
            test_indices.extend(idxs[:samples_per_class])
    else:
        test_indices = list(range(len(test_folder)))

    test_dataset = MedicalAnomalyDataset(
        Subset(test_folder, test_indices), is_train=False, normal_class_idx=test_normal_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 0 workers for evaluation eliminates multiprocessing shm crashes / SIGABRT
        pin_memory=False,
    )

    return {"train": train_loader, "test": test_loader}
