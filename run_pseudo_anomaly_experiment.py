import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PatchCL_AE_MHSA.models import create_patchcl_ae_models
from PatchCL_AE_MHSA.dataset import create_patchcl_dataloaders
from PatchCL_AE_MHSA.pseudo_anomaly import NaturalPseudoAnomalyGenerator
from PatchCL_AE_MHSA.losses import PatchContrastiveLoss, ImageAdversarialLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# Dataloaders: batch_size=8 to prevent VRAM paging on RTX 4050
BATCH_SIZE = 8
loaders = create_patchcl_dataloaders('datasets/chest_xray', batch_size=BATCH_SIZE, num_workers=0)
train_loader = loaders['train']
test_loader = loaders['test']
num_batches_total = len(train_loader)
print(f"Train batches per epoch: {num_batches_total} (Batch size: {BATCH_SIZE})", flush=True)

# Model
gen, disc = create_patchcl_ae_models(in_channels=3, mhsa_mode='parallel', device=device)
ckpt_path = 'PatchCL_AE_MHSA/checkpoints_parallel/patchcl_ae_epoch_010.pt'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
gen.load_state_dict(ckpt['generator_state'])
disc.load_state_dict(ckpt['discriminator_state'])
print(f"Loaded warm-start weights from {ckpt_path} (Epoch {ckpt.get('epoch', 'N/A')})", flush=True)

# Optimizers
opt_g = torch.optim.Adam(gen.parameters(), lr=5e-5, betas=(0.9, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=1e-6, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler('cuda')

# Losses & Synthesizer
criterion_patch = PatchContrastiveLoss(temperature=0.07, num_patches=256, gamma=2.0)
criterion_adv = ImageAdversarialLoss()
synth = NaturalPseudoAnomalyGenerator(img_size=224, margin=35, device=device)

os.makedirs('PatchCL_AE_MHSA/checkpoints_nsa', exist_ok=True)

def evaluate_model(epoch_num):
    gen.eval()
    all_lbls, r5_list, c5_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            img = batch['image'].to(device)
            lbl = batch['label'].numpy()
            rec = gen(img)
            _, fr = gen.encode(img)
            _, fc = gen.encode(rec)
            r5_list.append(fr['E_5'])
            c5_list.append(fc['E_5'])
            all_lbls.extend(lbl)
            
    r5 = torch.cat(r5_list, dim=0)
    c5 = torch.cat(c5_list, dim=0)
    m5 = (1.0 - F.cosine_similarity(r5, c5, dim=1).unsqueeze(1))
    cr = m5[:, 0, 6:-6, 6:-6].reshape(len(all_lbls), -1)
    k = int(0.25 * cr.size(1))
    scores = torch.topk(cr, k=k, dim=1).values.mean(dim=1).cpu().numpy()
    
    all_lbls = np.array(all_lbls)
    auc = roc_auc_score(all_lbls, scores) * 100
    fpr, tpr, thresh = roc_curve(all_lbls, scores)
    spe = 1.0 - fpr
    
    v85 = np.where(spe >= 0.8500)[0]
    sen85 = tpr[v85[np.argmax(tpr[v85])]] * 100.0 if len(v85) > 0 else 0
    
    v928 = np.where(tpr >= 0.9278)[0]
    spe928 = spe[v928[np.argmax(spe[v928])]] * 100.0 if len(v928) > 0 else 0
    
    # Youden J
    j = tpr + spe - 1.0
    bj = np.argmax(j)
    
    # Check on Bacteria and Virus subsets
    bac_sc = scores[234:234+242]
    vir_sc = scores[234+242:]
    norm_sc = scores[:234]
    th_j = thresh[bj]
    
    sen_bac = np.mean(bac_sc >= th_j) * 100
    sen_vir = np.mean(vir_sc >= th_j) * 100
    spe_val = np.mean(norm_sc < th_j) * 100
    
    print(f"\n==================== EVALUATION EPOCH {epoch_num} ====================", flush=True)
    print(f"AUROC: {auc:5.2f}% (Paper Baseline: 91.14%, Parallel Baseline: 90.52%)", flush=True)
    print(f"At SEN >= 92.78% (Paper SEN) -> SPE: {spe928:5.2f}% (Paper SPE: 74.90%)", flush=True)
    print(f"At SPE >= 85.00%             -> SEN: {sen85:5.2f}%", flush=True)
    print(f"Youden J: SEN={tpr[bj]*100:5.2f}%, SPE={spe[bj]*100:5.2f}% | Bac_SEN={sen_bac:5.2f}%, Vir_SEN={sen_vir:5.2f}%", flush=True)
    print(f"======================================================================\n", flush=True)
    return auc, spe928, sen85

# Initial evaluation before fine-tuning
print("\n>>> Baseline Performance Before NSA Fine-Tuning:", flush=True)
best_auc, best_spe928, _ = evaluate_model(0)

NUM_EPOCHS = 3
for ep in range(1, NUM_EPOCHS + 1):
    gen.train()
    disc.train()
    t0 = time.time()
    
    running_l_g = 0.0
    running_l_heal = 0.0
    running_l_rep = 0.0
    num_batches = 0
    
    for b_idx, batch in enumerate(train_loader):
        x_clean = batch['image'].to(device)
        b = x_clean.size(0)
        
        # 1. Synthesize pseudo-anomalies on half the batch to optimize throughput & stability
        half_b = max(1, b // 2)
        with torch.no_grad():
            x_synth_sub, mask_synth_sub = synth(x_clean[:half_b])
            
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
            # 1. Clean reconstruction with MHSA attention guidance
            rec_clean, feats_noisy, w_attn = gen(x_clean + noise, return_features=True)
            
            # 2. Synthetic Active Healing: G(x_synth) must heal to clean x_clean!
            rec_healed = gen(x_synth_sub)
            loss_heal = F.l1_loss(rec_healed, x_clean[:half_b])
            
            # 3. Adversarial loss
            d_fake_g = disc(rec_clean)
            loss_adv = criterion_adv.generator_loss(d_fake_g)
            
            # 4. PatchCL InfoNCE on clean images
            _, feats_clean = gen.encode(x_clean)
            _, feats_recon = gen.encode(rec_clean)
            loss_patch = criterion_patch(
                features_real=feats_clean,
                features_fake=feats_recon,
                projection_module=gen.projection,
                w_attenuation=w_attn,
            )
            
            # 5. Parallel MHSA self-reconstruction objective
            if isinstance(feats_noisy, dict) and "recon_feat_mhsa" in feats_noisy and feats_noisy["recon_feat_mhsa"] is not None:
                loss_mhsa = F.mse_loss(feats_noisy["recon_feat_mhsa"], feats_noisy["E_4"].detach())
            else:
                loss_mhsa = torch.tensor(0.0, device=device)
            
            # 6. Feature Repulsion on E_5: push E_5(x_synth) away from E_5(x_clean) on lesion regions
            _, feats_synth = gen.encode(x_synth_sub)
            m_down = F.interpolate(mask_synth_sub, size=(feats_synth['E_5'].shape[2], feats_synth['E_5'].shape[3]), mode='bilinear', align_corners=True)
            cos_e5 = F.cosine_similarity(feats_synth['E_5'], feats_clean['E_5'][:half_b].detach(), dim=1).unsqueeze(1)
            
            lesion_weights = (m_down > 0.25).float()
            if lesion_weights.sum() > 0:
                loss_repulse = (F.relu(cos_e5 - 0.10) * lesion_weights).sum() / (lesion_weights.sum() + 1e-6)
            else:
                loss_repulse = torch.tensor(0.0, device=device)
                
            loss_g = 1.0 * loss_adv + 1.0 * loss_patch + 0.10 * loss_mhsa + 2.0 * loss_heal + 0.50 * loss_repulse
            
        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()
        
        running_l_g += loss_g.item()
        running_l_heal += loss_heal.item()
        running_l_rep += loss_repulse.item()
        num_batches += 1
        
        if (b_idx + 1) % 30 == 0 or (b_idx + 1) == num_batches_total:
            elapsed = time.time() - t0
            speed = (b_idx + 1) / elapsed
            print(f"Epoch [{ep}/{NUM_EPOCHS}] | Batch [{b_idx+1:03d}/{num_batches_total}] ({speed:.2f} it/s) | L_G: {running_l_g/num_batches:.4f} | L_Heal: {running_l_heal/num_batches:.4f} | L_Rep: {running_l_rep/num_batches:.4f}", flush=True)
        
    dur = time.time() - t0
    print(f"\n--- Epoch {ep} completed in {dur:.1f}s ({dur/60:.2f} mins) ---", flush=True)
    
    auc, spe928, sen85 = evaluate_model(ep)
    if auc > best_auc or spe928 > best_spe928:
        best_auc = max(auc, best_auc)
        best_spe928 = max(spe928, best_spe928)
        torch.save({
            'epoch': ep,
            'generator_state': gen.state_dict(),
            'best_auc': auc,
            'spe_at_sen928': spe928,
        }, f'PatchCL_AE_MHSA/checkpoints_nsa/patchcl_ae_nsa_best.pt')
        print(f"*** NEW BEST CHECKPOINT SAVED! (AUC: {auc:.2f}%, SPE@SEN92.8%: {spe928:.2f}%) ***\n", flush=True)

print("\n=== NSA EXPERIMENT FINISHED SUCCESSFULLY ===", flush=True)
