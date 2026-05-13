import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from typing import Dict
from collections import Counter
import numpy as np
import cv2
from PIL import Image

# class MedianBlurTransform(object):
#     def __init__(self, kernel_size=3):
#         # Lưu ý: kernel_size phải là số lẻ (3, 5, 7...)
#         self.kernel_size = kernel_size

#     def __call__(self, img):
#         # 1. Chuyển PIL Image sang Numpy để dùng OpenCV
#         img_np = np.array(img)
        
#         # 2. Áp dụng Median Filter
#         # Hàm này trả về một mảng numpy.ndarray
#         img_blurred = cv2.medianBlur(img_np, self.kernel_size)
        
#         # 3. Chuyển ngược lại PIL Image bằng Image.fromarray
#         # Đã loại bỏ đoạn from_numpy gây lỗi
#         return Image.fromarray(img_blurred)

class OCTDataset(Dataset):
    def __init__(self, subset_dataset, is_train: bool, normal_folder_idx: int):
        self.dataset = subset_dataset
        self.is_train = is_train
        self.normal_folder_idx = normal_folder_idx

    def __getitem__(self, index):
        original_idx = self.dataset.indices[index]
        image, original_label = self.dataset.dataset[original_idx]
        img_path, _ = self.dataset.dataset.samples[original_idx]
        
        # LOGIC GÁN NHÃN:
        # Nếu original_label khớp với index của folder NORMAL -> nhãn 0
        # Ngược lại (là các bệnh lý) -> nhãn 1
        final_label = 0 if original_label == self.normal_folder_idx else 1
        
        return {
            "image": image,
            "label": final_label,
            "path": img_path  
        }

    def __len__(self):
        return len(self.dataset)

def create_oct_dataloaders(config: Dict):
    ds_cfg = config.get("dataset", {})
    train_cfg = config.get("train", {})
    
    root_path = ds_cfg["root"]
    img_size = ds_cfg.get("image_size", 256)
    normal_name = ds_cfg.get("normal_class_name", "NORMAL")
    
    max_train = ds_cfg.get("max_train_images")
    max_test = ds_cfg.get("max_test_images")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ds_cfg.get("normalize_mean", [0.5]*3), 
                             std=ds_cfg.get("normalize_std", [0.5]*3))
    ])

    # --- 1. XỬ LÝ TẬP TEST (LẤY ĐỀU CÁC LỚP) ---
    test_path = os.path.join(root_path, ds_cfg["test_dir"])
    test_set_full = datasets.ImageFolder(root=test_path, transform=transform)
    
    test_normal_idx = test_set_full.class_to_idx.get(normal_name)
    all_classes = test_set_full.classes
    num_classes = len(all_classes)

    if max_test:
        print(f"📊 Đang lấy mẫu đều {max_test} ảnh từ {num_classes} lớp...")
        indices_per_class = {i: [] for i in range(num_classes)}
        
        # Phân loại index theo từng lớp
        for idx, (_, label) in enumerate(test_set_full.imgs):
            indices_per_class[label].append(idx)
        
        # Tính số lượng ảnh cần lấy cho mỗi lớp
        samples_per_class = int(max_test) // num_classes
        
        test_indices = []
        for label, indices in indices_per_class.items():
            # Lấy mẫu từ mỗi lớp (nếu lớp đó ít ảnh hơn samples_per_class thì lấy hết)
            selected = indices[:samples_per_class]
            test_indices.extend(selected)
            print(f"  + Lớp '{all_classes[label]}': lấy {len(selected)} ảnh")
    else:
        test_indices = list(range(len(test_set_full)))

    test_dataset = OCTDataset(Subset(test_set_full, test_indices), is_train=False, normal_folder_idx=test_normal_idx)

    # --- 2. XỬ LÝ TẬP TRAIN (CHỈ LẤY NORMAL) ---
    train_path = os.path.join(root_path, ds_cfg["train_dir"])
    train_set_full = datasets.ImageFolder(root=train_path, transform=transform)
    
    # Tìm index của folder NORMAL trong tập Train
    train_normal_idx = train_set_full.class_to_idx.get(normal_name)
    
    # Lọc chỉ lấy ảnh NORMAL
    train_indices = [i for i, (_, lbl) in enumerate(train_set_full.imgs) if lbl == train_normal_idx]
    print(train_indices[:5])  # Debug: in ra 5 index đầu tiên của ảnh NORMAL trong tập Train
    if max_train:
        train_indices = train_indices[:int(max_train)]
    
    # Tập Train cũng dùng logic gán nhãn tương tự (nhưng vì chỉ có Normal nên tất cả sẽ là 0)
    train_dataset = OCTDataset(Subset(train_set_full, train_indices), is_train=True, normal_folder_idx=train_normal_idx)

    # --- 3. DATALOADERS ---
    bs = 16 if train_cfg.get("batch_size") == "auto" else train_cfg.get("batch_size", 16)
    nw = 4 if ds_cfg.get("num_workers") == "auto" else ds_cfg.get("num_workers", 4)
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # --- 4. KIỂM TRA (DEBUG) ---
    print("\n" + "="*50)
    print("🧪 DATA INSPECTION (Normal=0, Disease=1)")
    
    # Kiểm tra Train
    t_batch = next(iter(train_loader))
    print(f"[TRAIN] Labels in batch: {torch.unique(t_batch['label']).tolist()} (Expect: [0])")

    # Kiểm tra Test
    all_test_labels = []
    for b in test_loader:
        all_test_labels.extend(b["label"].tolist())
    test_dist = Counter(all_test_labels)
    
    print(f"[TEST] Distribution:")
    print(f"  - Label 0 (Normal): {test_dist[0]} images")
    print(f"  - Label 1 (Diseases): {test_dist[1]} images")
    print("="*50 + "\n")

    return {"train": train_loader, "test": test_loader}