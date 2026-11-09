# PatchCL-AE (Cải tiến): Anomaly Detection for Medical Images via MHSA Bottleneck, Memory Bank & Attention-Guided Patch Contrastive Learning

Mô hình phát hiện bất thường trên ảnh y tế (OCT, X-quang, MRI), được kế thừa và phát triển từ kiến trúc nền tảng **PatchCL-AE**:
> **Shuai Lu, Weihang Zhang, Jia Guo, Hanruo Liu, Huiqi Li, Ningli Wang**  
> *"PatchCL-AE: Anomaly detection for medical images using patch-wise contrastive learning-based auto-encoder"*  
> Computerized Medical Imaging and Graphics 114 (2024) 102366.  
> DOI: [10.1016/j.compmedimag.2024.102366](https://doi.org/10.1016/j.compmedimag.2024.102366)

Toàn bộ phiên bản mô hình chuẩn ban đầu đã được sao lưu nguyên vẹn tại thư mục [`PatchCL_AE_original/`](../PatchCL_AE_original).

---

## 1. Các đóng góp khoa học chính (Main Contributions)

1. **Khắc phục xung đột mẫu âm tính giả (Attention-Guided Negative Filtering):**
   * *Hạn chế của PatchCL-AE gốc:* Giả định mọi patch ở vị trí khác nhau đều là mẫu âm tính (Negative). Trên ảnh y tế có tính đối xứng giải phẫu hoặc vùng nhu mô đồng nhất rộng lớn, giả định này gây ra **xung đột mẫu âm tính giả (False Negative Semantic Collision)**, làm suy giảm khả năng phân biệt đặc trưng.
   * *Giải pháp:* Tận dụng ma trận chú ý đối xứng chuẩn hóa $S$ từ khối MHSA để xác định mức độ liên kết ngữ nghĩa giữa các patch. Những vị trí có mức độ chú ý cao sẽ được giảm nhẹ hình phạt mẫu âm trong hàm InfoNCE thông qua trọng số phạt $w(s, s') = (1 - S_{s, s'})^\gamma$.

2. **Nắm bắt ngữ cảnh giải phẫu toàn cục (Global Context Modeling via MHSA Bottleneck):**
   * Thay thế 6 khối tích chập cục bộ tại bottleneck ($E_5$) bằng **Multi-Head Self-Attention (MHSA) Transformer Bottleneck**, giúp mạng nắm bắt các phụ thuộc tầm xa (long-range dependencies) và cấu trúc hình thái giải phẫu tổng thể.

3. **Ngăn chặn triệt để hiện tượng rò rỉ tổn thương (Pathology-Constrained Reconstruction via Memory Bank):**
   * Tích hợp **Memory Bank** chứa $K = 800$ nguyên mẫu mô lành tính (normal prototypes) vào giữa MHSA bottleneck và Decoder. Ép Decoder chỉ được tái tạo ảnh từ tổ hợp thưa (Top-$p$ Sparsity, $p = 3$) của các nguyên mẫu lành có trong bộ nhớ, triệt tiêu khả năng học ánh xạ đồng nhất (identity mapping) và biến đổi vùng bệnh lý thành mô lành ở ảnh đầu ra.

---

## 2. Cấu trúc thư mục

```text
PatchCL_AE/
├── __init__.py           # Export toàn bộ modules, models, losses, metrics, trainers
├── models.py             # Kiến trúc cải tiến: Encoder (E_1..E_4), MHSABottleneck, MemoryBank, Decoder, Discriminator, ProjectionHead
├── losses.py             # ImageAdversarialLoss, AttentionWeightedPatchContrastiveLoss, PatchCL_TotalLoss
├── anomaly_score.py      # Bản đồ bất thường đa tầng M = sum resize(M_l) từ E_1 đến E_4
├── metrics.py            # Tính AUC, best F1-score, Accuracy, Sensitivity, Specificity
├── dataset.py            # DataLoader cho tập dữ liệu (chỉ lấy NORMAL khi train, đầy đủ các lớp khi test)
├── trainer.py            # Bộ huấn luyện hoàn chỉnh tích hợp AMP, Attention-Guided InfoNCE và Memory Losses
├── train.py              # CLI script huấn luyện với các siêu tham số MHSA, Memory Bank, Loss weights
├── eval.py               # CLI script đánh giá checkpoint & xuất visualization heatmaps
├── improve.markdown      # Tài liệu đặc tả yêu cầu cải tiến khoa học
├── checkpoints/          # Thư mục lưu trọng số model
└── logs/                 # Thư mục lưu nhật ký huấn luyện
```

---

## 3. Luồng dữ liệu và chi tiết kiến trúc (Dataflow & Architecture)

### 3.1. Sơ đồ luồng dữ liệu tổng thể

```text
Ảnh gốc X (256×256)
   │
   ▼
[ Encoder: E_1 → E_4 ] ──(Đặc trưng đa tầng)──► [ Projection Heads P_1 → P_4 ]
   │                                                     │
   ▼                                                     ▼
[ Bottleneck: MHSA Block ] ──── Attention Map S ────► [ Attention-Guided InfoNCE ]
   │ (Feature Map Z)          (Đã ngắt gradient)           (Tại E_3, E_4)
   ▼
[ MEMORY BANK MODULE ] ◄── (Học trên các nguyên mẫu mô lành)
   │ (Feature Map Ẑ)
   ▼
[ Decoder: De_4 → De_1 ]
   │
   ▼
Ảnh tái tạo X̂ (256×256) ──► [ Discriminator D ] & [ Projection Heads P_1 → P_4 ]
```

### 3.2. Chi tiết từng khối kiến trúc

* **Encoder ($E_1 \to E_4$):**
  * `E_1`: $224 \times 224$ | Conv $7 \times 7$, 64 kênh, InstanceNorm, ReLU
  * `E_2`: $112 \times 112$ | Conv $3 \times 3$, 128 kênh, InstanceNorm, ReLU, AvgPool $2 \times 2$
  * `E_3`: $56 \times 56$   | Conv $3 \times 3$, 256 kênh, InstanceNorm, ReLU, AvgPool $2 \times 2$
  * `E_4`: $28 \times 28$   | Conv $3 \times 3$, 256 kênh, InstanceNorm, ReLU, AvgPool $2 \times 2$

* **MHSA Transformer Bottleneck (Thay thế khối $E_5$ cũ):**
  * Đầu vào: Feature map $Z \in \mathbb{R}^{B \times 256 \times 28 \times 28}$ ($N = 28 \times 28 = 784$ tokens).
  * 2 khối Transformer Bottleneck (`MHSABottleneckBlock`), 8 attention heads ($d_k = 32$), learnable 2D spatial positional embeddings, FFN tỉ lệ $4\times$ ($256 \to 1024 \to 256$) với GELU và Pre-LayerNorm.
  * Xuất ra feature map ngữ cảnh toàn cục và ma trận chú ý đối xứng $S \in \mathbb{R}^{B \times 784 \times 784}$.

* **Memory Bank Module (Nằm giữa MHSA Bottleneck và Decoder $De_4$):**
  * Bộ nhớ $\mathbf{M} \in \mathbb{R}^{K \times d}$ với $K = 800$ nguyên mẫu mô lành, $d = 256$ kênh.
  * Truy vấn Cosine Similarity nhiệt độ $\tau_m = 0.1$.
  * Lọc thưa Top-$p$ Sparsity với $p = 3$: chỉ giữ lại 3 ô nhớ có trọng số cao nhất, gán phần còn lại bằng 0 và chuẩn hóa lại.
  * Tạo đặc trưng tái tạo lành tính: $\hat{Z} \in \mathbb{R}^{B \times 256 \times 28 \times 28}$.

* **Decoder ($De_4 \to De_1$):**
  * `De_4`: $56 \times 56$   | Bilinear Upsample ($\times 2$), Conv $3 \times 3$, 256 kênh, InstanceNorm, ReLU
  * `De_3`: $112 \times 112$ | Bilinear Upsample ($\times 2$), Conv $3 \times 3$, 128 kênh, InstanceNorm, ReLU
  * `De_2`: $224 \times 224$ | Bilinear Upsample ($\times 2$), Conv $3 \times 3$, 64 kênh, InstanceNorm, ReLU
  * `De_1`: $224 \times 224$ | Conv $7 \times 7$, 3 kênh, Tanh
  * **Tuyệt đối không dùng Skip Connection** từ Encoder sang Decoder để bảo toàn cơ chế chặn tổn thương của Memory Bank.

* **Discriminator ($D_1 \to D_9$):**
  * Kiến trúc GAN Discriminator 9 tầng: $D_1 \dots D_6$ là Conv $3 \times 3$ với InstanceNorm + LeakyReLU (0.2) giảm dần từ $112 \times 112$ về $4 \times 4$, sau đó Flatten qua $D_7$ (8192d) $\to$ $D_8$ (256d) $\to$ $D_9$ (1 logit).

* **Projection Heads ($P_1 \to P_4$):**
  * Các đầu chiếu MLP 2 lớp riêng biệt cho 4 tầng $E_1, E_2, E_3, E_4$ (kích thước kênh tương ứng $64, 128, 256, 256 \to 256 \to 256$).

---

## 4. Hệ thống công thức toán học chi tiết (Loss Formulations)

### 4.1. Attention-Weighted Patch-wise Contrastive Loss ($\mathcal{L}_{Patch\_Attn}$)

Từ khối MHSA ở bottleneck, ma trận chú ý đối xứng chuẩn hóa được tính như sau:

$$\bar{A} = \frac{1}{H} \sum_{h=1}^H \text{Softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right), \quad S = \frac{\bar{A} + \bar{A}^T}{2}$$

Trọng số phạt mẫu âm giữa vị trí $s$ và $s'$:

$$w(s, s') = (1 - S_{s, s'})^\gamma \quad (\text{với } \gamma = 1.0)$$

Hàm mất mát InfoNCE có trọng số chú ý tại vị trí lấy mẫu $s$:

$$\mathcal{L}_{\text{Patch\_Attn}}(s) = -\log \frac{\exp\left(\frac{\text{sim}(\hat{z}^s, z^{s+})}{\tau}\right)}{\exp\left(\frac{\text{sim}(\hat{z}^s, z^{s+})}{\tau}\right) + \sum_{s' \neq s} w(s, s') \cdot \exp\left(\frac{\text{sim}(\hat{z}^s, z^{s'-})}{\tau}\right)}$$

* **Cơ chế áp dụng theo tầng:**
  * Tầng $E_4$ ($28 \times 28$): Tọa độ mẫu trực tiếp tương ứng ma trận chú ý $S$ ($784 \times 784$).
  * Tầng $E_3$ ($56 \times 56$): Tọa độ mẫu $(r, c)$ được ánh xạ về $(r // 2, c // 2)$ trên grid $28 \times 28$ của $S$.
  * Tầng $E_1, E_2$: Áp dụng InfoNCE chuẩn ($w(s, s') = 1.0$).
* Hàm loss được tính trung bình trên các vị trí lấy mẫu $s \in S_l$ ($K = 256$ vị trí) và các tầng $l \in \{E_1, E_2, E_3, E_4\}$.

### 4.2. Các hàm mất mát cho Memory Bank ($\mathcal{L}_{Mem}$)

Với mỗi token $z_i$ ($i = 1 \dots 784$):
* **Trọng số truy vấn:**
  $$w_{i, k} = \frac{\exp\left(\text{sim}(z_i, m_k) / \tau_m\right)}{\sum_{j=1}^K \exp\left(\text{sim}(z_i, m_j) / \tau_m\right)} \quad (\tau_m = 0.1)$$
* **Lọc thưa Top-$p$ Sparsity:** Giữ lại $p = 3$ trọng số lớn nhất, gán phần còn lại bằng 0 và chuẩn hóa lại thành $\hat{w}_{i, k}$.
* **Đặc trưng tái tạo:** $\hat{z}_i = \sum_{k=1}^K \hat{w}_{i, k} m_k$.

Để tối ưu hóa Memory Bank, bổ sung 3 thành phần mất mát:
1. **Compactness Loss (Ép đặc trưng mô lành bám sát ô nhớ gần nhất):**
   $$\mathcal{L}_{compact} = \frac{1}{N} \sum_{i=1}^N \|z_i - m_{\text{nearest}}\|_2^2 \quad (\text{với } m_{\text{nearest}} \text{ được detach})$$
2. **Separateness Loss (Đẩy các ô nhớ ra xa nhau, tránh suy biến/trùng lặp):**
   $$\mathcal{L}_{sep} = \frac{1}{K(K-1)} \sum_{j \neq k} \max\left(0, \delta - \|m_j - m_k\|_2\right) \quad (\text{với margin } \delta = 0.1)$$
3. **Entropy Loss (Ép phân phối truy vấn thưa và tập trung):**
   $$\mathcal{L}_{ent} = - \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K w_{i, k} \log(w_{i, k} + 10^{-12})$$

$$\mathcal{L}_{Mem} = \mathcal{L}_{compact} + 0.1 \cdot \mathcal{L}_{sep} + 0.0001 \cdot \mathcal{L}_{ent}$$

### 4.3. Tổng hợp hàm mất mát Generator cân bằng ($\mathcal{L}_{G}$)

Để kiểm soát sự áp đảo của GAN Discriminator và giúp các nhánh $\mathcal{L}_{Patch\_Attn}$ và $\mathcal{L}_{Mem}$ phát huy tối đa tác dụng, hàm mất mát của Generator được cân bằng theo công thức:

$$\mathcal{L}_{G} = \lambda_{adv} \mathcal{L}_{Adv} + \lambda_{patch} \mathcal{L}_{Patch\_Attn} + \lambda_{mem} \mathcal{L}_{Mem}$$

* Mặc định: $\lambda_{adv} = 0.1$, $\lambda_{patch} = 1.0$, $\lambda_{mem} = 0.1$.
* Việc giảm hệ số của $\mathcal{L}_{Adv}$ xuống 0.1 đưa đóng góp thực tế của nó về mức xấp xỉ $\sim 0.6 - 0.8$, tương đương với $\mathcal{L}_{Patch}$ và $\mathcal{L}_{Mem}$, giúp các gradient truyền ngược đồng đều mà không bị Adversarial Loss nuốt chửng.

---

## 5. Những lưu ý kỹ thuật chống rò rỉ tổn thương (Lesion Leakage)

1. **Tuyệt đối không dùng Skip Connections từ Encoder sang Decoder:**
   * Luồng dữ liệu bắt buộc phải thắt cổ chai hoàn toàn qua Memory Bank để đảm bảo mọi vùng bệnh lý bị ép thay thế bằng tổ hợp mô lành.
2. **Bắt buộc dùng cơ chế lọc thưa (Top-$p$ Sparsity, $p = 3$):**
   * Giới hạn $p$ rất nhỏ ($p \in [3, 5]$) ngăn chặn mạng nơ-ron kết hợp tuyến tính nhiều ô nhớ khác nhau với tỉ lệ nhỏ để "vẽ lại" mảng bệnh lý bất thường.
3. **Ngắt Gradient (`.detach()`) trên Attention Map khi đưa vào InfoNCE:**
   * Chống hiện tượng shortcut learning: nếu không ngắt gradient, mạng sẽ tự động đẩy $S \to 1$ để triệt tiêu mẫu số của InfoNCE về 0 nhằm giảm loss dễ dãi.
4. **Kích thước Memory Bank tối ưu ($K = 800$):**
   * Đủ bao phủ các biến thể giải phẫu bình thường mà không gây dư thừa dung lượng khiến mô hình ghi nhớ cả nhiễu bất thường.

---

## 6. Hướng dẫn sử dụng

### 6.1. Sử dụng trong Python Script

```python
import torch
from PatchCL_AE import (
    create_patchcl_ae_models,
    compute_patchcl_anomaly_map,
    anomaly_map_to_score,
    PatchCLTrainer,
    create_patchcl_dataloaders,
)

# 1. Khởi tạo mô hình cải tiến với MHSA Bottleneck và Memory Bank
generator, discriminator = create_patchcl_ae_models(
    num_mhsa_layers=2,
    num_heads=8,
    num_prototypes=800,
    top_p=3,
    tau_m=0.1,
    alpha_sep=0.1,
    beta_ent=0.0001,
    delta_sep=0.1,
)

# 2. Tạo DataLoaders (Train chỉ lấy ảnh Normal, Test lấy cả Normal và Abnormal)
loaders = create_patchcl_dataloaders(
    root_path="datasets/OCT2017",
    img_size=224,
    resize_size=256,
    batch_size=16,
)

# 3. Huấn luyện mô hình
trainer = PatchCLTrainer(
    generator=generator,
    discriminator=discriminator,
    train_loader=loaders["train"],
    test_loader=loaders["test"],
    lr=0.002,
    lambda_mem=0.1,
    gamma=1.0,
    use_amp=True,
)
trainer.fit(epochs=400, eval_every=5, save_every=10)

# 4. Tính toán Anomaly Map trong giai đoạn kiểm thử
generator.eval()
test_img = torch.randn(1, 3, 224, 224)
anomaly_map, recon_img, layer_maps = compute_patchcl_anomaly_map(generator, test_img)
score = anomaly_map_to_score(anomaly_map, reduction="max")
print(f"Anomaly Score: {score.item():.4f}")
```

### 6.2. Huấn luyện qua giao diện dòng lệnh (CLI)

```powershell
# Chạy huấn luyện đầy đủ 400 epochs với cấu hình chuẩn
.\.venv\Scripts\python.exe -m PatchCL_AE.train `
    --data_root datasets/OCT2017 `
    --epochs 400 `
    --batch_size 16 `
    --lr 0.002 `
    --num_mhsa_layers 2 `
    --num_heads 8 `
    --num_prototypes 800 `
    --top_p 3 `
    --tau_m 0.1 `
    --lambda_mem 0.1 `
    --gamma 1.0 `
    --checkpoint_dir PatchCL_AE/checkpoints `
    --log_dir PatchCL_AE/logs
```

### 6.3. Đánh giá Checkpoint và xuất Heatmap Visualization

```powershell
# Đánh giá checkpoint tốt nhất và xuất các bản đồ nhiệt trực quan
.\.venv\Scripts\python.exe -m PatchCL_AE.eval `
    --checkpoint PatchCL_AE/checkpoints/patchcl_ae_best_auc.pt `
    --data_root datasets/OCT2017 `
    --batch_size 16 `
    --reduction max `
    --viz_dir visualizations/patchcl_ae `
    --max_viz 20
```

---

## 7. Các tham số dòng lệnh chính (`train.py`)

| Tham số | Mặc định | Ý nghĩa |
| :--- | :---: | :--- |
| `--data_root` | `datasets/OCT2017` | Đường dẫn tới thư mục gốc của bộ dữ liệu |
| `--epochs` | `400` | Số epoch huấn luyện tối đa |
| `--batch_size` | `16` | Kích thước batch (khuyến nghị 8 nếu VRAM < 6GB) |
| `--lr` | `0.002` | Tốc độ học ban đầu cho Adam |
| `--num_mhsa_layers` | `2` | Số khối MHSA Transformer tại Bottleneck |
| `--num_heads` | `8` | Số attention heads trong MHSA Bottleneck |
| `--num_prototypes` | `800` | Dung lượng $K$ nguyên mẫu lành trong Memory Bank |
| `--top_p` | `3` | Tham số lọc thưa Top-$p$ cho Memory Bank |
| `--tau_m` | `0.1` | Nhiệt độ truy vấn $\tau_m$ của Memory Bank |
| `--lambda_adv` | `0.1` | Trọng số $\lambda_{adv}$ cho Adversarial Loss (giảm xuống 0.1 để cân bằng GAN) |
| `--lambda_patch` | `1.0` | Trọng số $\lambda_{patch}$ cho Attention-Guided InfoNCE Loss |
| `--lambda_mem` | `0.1` | Trọng số $\lambda_{mem}$ cho hàm mất mát Memory Bank |
| `--alpha_sep` | `0.1` | Hệ số $\alpha$ cho Separateness Loss |
| `--beta_ent` | `0.0001` | Hệ số $\beta$ cho Entropy Loss |
| `--delta_sep` | `0.1` | Khoảng cách lề $\delta$ cho Separateness Loss |
| `--gamma` | `1.0` | Số mũ $\gamma$ cho trọng số Attention-Guided InfoNCE |
| `--reduction` | `mean` | Phương pháp quy đổi bản đồ nhiệt sang điểm ảnh (`mean`, `top_k`, `max`, `top_percent`) |
| `--top_k` | `100` | Số pixel lớn nhất để lấy trung bình khi `--reduction top_k` |
| `--no_amp` | `False` | Cờ tắt chế độ Automatic Mixed Precision (mặc định AMP luôn được **BẬT** để tăng tốc) |
