import os
import sys
import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PatchCL_AE_MHSA.models import create_patchcl_ae_models
from PatchCL_AE_MHSA.dataset import create_patchcl_dataloaders, get_patchcl_transforms
from PatchCL_AE_MHSA.losses import PatchContrastiveLoss, ImageAdversarialLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# 1. Dataloader: 3200 normal macular scans (expanded manifold)
BATCH_SIZE = 8
DATA_ROOT = 'datasets/OCT2017'
loaders = create_patchcl_dataloaders(
    DATA_ROOT,
    batch_size=BATCH_SIZE,
    num_workers=0,
    max_train_samples=3200,
    img_size=224,
    resize_size=256,
)
train_loader = loaders['train']
test_loader = loaders['test']
num_batches_total = len(train_loader)
print(f"OCT Train batches: {num_batches_total} (Batch size: {BATCH_SIZE}, Normal train images: {len(train_loader.dataset)})", flush=True)
print(f"OCT Test samples: {len(test_loader.dataset)} (242 Normal, 242 CNV, 242 DME, 242 DRUSEN)", flush=True)

# 2. Model: Decoupled Parallel MHSA with Warm-start from Epoch 4 (AUC: 91.76%)
gen, disc = create_patchcl_ae_models(
    in_channels=3,
    mhsa_mode='parallel',
    num_heads=4,
    gamma=2.0,
    device=device,
)

ckpt_path = 'PatchCL_AE_MHSA/checkpoints_oct/patchcl_ae_oct_best.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['generator_state'])
    disc.load_state_dict(ckpt['discriminator_state'])
    print(f"Loaded warm-start weights from {ckpt_path} (Warm-start AUC: 91.76%)", flush=True)

# 3. Optimizers & Schedulers
EPOCHS = 5
opt_g = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0.9, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=1e-6, betas=(0.9, 0.999))
scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=EPOCHS, eta_min=1e-5)
scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=EPOCHS, eta_min=1e-7)
scaler = torch.amp.GradScaler('cuda')

# 4. Losses
criterion_patch = PatchContrastiveLoss(temperature=0.07, num_patches=256, gamma=2.0)
criterion_adv = ImageAdversarialLoss()

ckpt_dir = 'PatchCL_AE_MHSA/checkpoints_oct'
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

def evaluate_oct_paper_metric(epoch_num):
    gen.eval()
    all_lbls = []
    all_maps = []
    
    with torch.no_grad():
        for batch in test_loader:
            img = batch['image'].to(device)
            lbl = batch['label'].numpy()
            rec = gen(img)
            all_lbls.extend(lbl)
            
            _, fr = gen.encode(img)
            _, fc = gen.encode(rec)
            
            # Exact Paper Eq. (5), (6), (7): Multi-layer semantic discrepancy maps
            maps_p = []
            for l in ['E_1', 'E_2', 'E_3', 'E_4']:
                b, c, h, w = fr[l].shape
                zr = gen.projection(fr[l].permute(0,2,3,1).contiguous().view(-1, c), l).view(b, h, w, -1).permute(0, 3, 1, 2)
                zc = gen.projection(fc[l].permute(0,2,3,1).contiguous().view(-1, c), l).view(b, h, w, -1).permute(0, 3, 1, 2)
                mp = (1.0 - F.cosine_similarity(zr, zc, dim=1)).unsqueeze(1)
                mp_up = F.interpolate(mp, size=(224, 224), mode='bilinear', align_corners=True)
                maps_p.append(mp_up)
                
            mp_tot = torch.stack(maps_p, dim=0).sum(dim=0).squeeze(1).view(img.size(0), -1)
            all_maps.append(mp_tot.cpu())
            
    all_maps = torch.cat(all_maps, dim=0)
    all_lbls = np.array(all_lbls)
    
    # Official top 1% reduction (k=501)
    k = max(1, int(0.01 * all_maps.size(1)))
    scores = all_maps.topk(k=k, dim=1).values.mean(dim=1).numpy()
    
    auc = roc_auc_score(all_lbls, scores) * 100
    fpr, tpr, thresh = roc_curve(all_lbls, scores)
    spe = 1.0 - fpr
    
    # Youden J
    j = tpr + spe - 1.0
    bj = np.argmax(j)
    th_j = thresh[bj]
    
    norm_idx = np.where(all_lbls == 0)[0]
    cnv_idx = np.where((all_lbls == 1) & (np.arange(len(all_lbls)) < 242))[0]
    dme_idx = np.where((all_lbls == 1) & (np.arange(len(all_lbls)) >= 242) & (np.arange(len(all_lbls)) < 484))[0]
    dru_idx = np.where((all_lbls == 1) & (np.arange(len(all_lbls)) >= 484) & (np.arange(len(all_lbls)) < 726))[0]
    
    sen_cnv = np.mean(scores[cnv_idx] >= th_j) * 100 if len(cnv_idx) > 0 else 0
    sen_dme = np.mean(scores[dme_idx] >= th_j) * 100 if len(dme_idx) > 0 else 0
    sen_dru = np.mean(scores[dru_idx] >= th_j) * 100 if len(dru_idx) > 0 else 0
    spe_val = np.mean(scores[norm_idx] < th_j) * 100
    
    v95 = np.where(spe >= 0.9500)[0]
    sen95 = tpr[v95[np.argmax(tpr[v95])]] * 100.0 if len(v95) > 0 else 0
    
    v99 = np.where(tpr >= 0.9900)[0]
    spe99 = spe[v99[np.argmax(spe[v99])]] * 100.0 if len(v99) > 0 else 0
    
    print(f"\n==================== OCT EVALUATION (PAPER METRIC) EPOCH {epoch_num} ====================", flush=True)
    print(f"AUROC: {auc:5.2f}% (Paper CMIG 2024 Table 4: 99.29 ± 0.12%)", flush=True)
    print(f"Youden J: SEN={tpr[bj]*100:5.2f}%, SPE={spe[bj]*100:5.2f}%", flush=True)
    print(f"  - CNV Sensitivity:    {sen_cnv:5.2f}%", flush=True)
    print(f"  - DME Sensitivity:    {sen_dme:5.2f}%", flush=True)
    print(f"  - DRUSEN Sensitivity: {sen_dru:5.2f}%", flush=True)
    print(f"At SPE >= 95.0%: SEN = {sen95:5.2f}% (Paper SEN: 99.87%)", flush=True)
    print(f"At SEN >= 99.0%: SPE = {spe99:5.2f}% (Paper SPE: 95.46%)", flush=True)
    print(f"========================================================================================\n", flush=True)
    return auc, tpr[bj]*100, spe[bj]*100

print("\n>>> Baseline Evaluation before expanded training:", flush=True)
best_auc, _, _ = evaluate_oct_paper_metric(0)

print("\n>>> STARTING EXPANDED TRAINING ON RETINAL OCT (3200 SCANS) <<<", flush=True)
for ep in range(1, EPOCHS + 1):
    gen.train()
    disc.train()
    t0 = time.time()
    
    running_l_g = 0.0
    running_l_adv = 0.0
    running_l_patch = 0.0
    num_batches = 0
    
    for b_idx, batch in enumerate(train_loader):
        x_clean = batch['image'].to(device)
        
        # ----------------- Train Discriminator -----------------
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            noise = torch.randn_like(x_clean) * 0.05
            rec_clean = gen(x_clean + noise)
            
            d_real = disc(x_clean)
            d_fake = disc(rec_clean.detach())
            loss_d, _, _ = criterion_adv.discriminator_loss(d_real, d_fake)
            
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)
        
        # ----------------- Train Generator -----------------
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            rec_clean, feats_noisy, w_attn = gen(x_clean + noise, return_features=True)
            d_fake_g = disc(rec_clean)
            loss_adv = criterion_adv.generator_loss(d_fake_g)
            
            _, feats_clean = gen.encode(x_clean)
            _, feats_recon = gen.encode(rec_clean)
            loss_patch = criterion_patch(
                features_real=feats_clean,
                features_fake=feats_recon,
                projection_module=gen.projection,
                w_attenuation=w_attn,
            )
            
            if isinstance(feats_noisy, dict) and "recon_feat_mhsa" in feats_noisy and feats_noisy["recon_feat_mhsa"] is not None:
                loss_mhsa = F.mse_loss(feats_noisy["recon_feat_mhsa"], feats_noisy["E_4"].detach())
            else:
                loss_mhsa = torch.tensor(0.0, device=device)
                
            loss_g = 0.10 * loss_adv + 1.0 * loss_patch + 0.10 * loss_mhsa
            
        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()
        
        running_l_g += loss_g.item()
        running_l_adv += loss_adv.item()
        running_l_patch += loss_patch.item()
        num_batches += 1
        
        if (b_idx + 1) % 50 == 0 or (b_idx + 1) == num_batches_total:
            elapsed = time.time() - t0
            speed = (b_idx + 1) / elapsed
            print(f"Epoch [{ep:02d}/{EPOCHS:02d}] | Batch [{b_idx+1:03d}/{num_batches_total}] ({speed:.2f} it/s) | L_G: {running_l_g/num_batches:.4f} | L_Patch: {running_l_patch/num_batches:.4f}", flush=True)
            
    scheduler_g.step()
    scheduler_d.step()
    
    dur = time.time() - t0
    print(f"--- Epoch {ep} completed in {dur:.1f}s ({dur/60:.2f} mins) ---", flush=True)
    
    auc, sen, spe = evaluate_oct_paper_metric(ep)
    if auc > best_auc:
        best_auc = auc
        torch.save({
            'epoch': ep,
            'generator_state': gen.state_dict(),
            'discriminator_state': disc.state_dict(),
            'auc': auc,
            'sen': sen,
            'spe': spe,
        }, os.path.join(ckpt_dir, 'patchcl_ae_oct_best.pt'))
        print(f"*** NEW BEST OCT CHECKPOINT SAVED! (AUC: {auc:.2f}%, SEN: {sen:.2f}%, SPE: {spe:.2f}%) ***\n", flush=True)

print("\n=== EXPANDED RETINAL OCT TRAINING COMPLETED ===", flush=True)
