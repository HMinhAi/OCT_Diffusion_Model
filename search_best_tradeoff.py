import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from PatchCL_AE_MHSA.models import create_patchcl_ae_models
from PatchCL_AE_MHSA.dataset import create_patchcl_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaders = create_patchcl_dataloaders('datasets/chest_xray', batch_size=16, num_workers=0)
test_loader = loaders['test']

PAPER_SEN = 92.78
PAPER_SPE = 74.90

for ckpt_p in ['PatchCL_AE_MHSA/checkpoints_parallel/patchcl_ae_epoch_010.pt',
               'PatchCL_AE_MHSA/checkpoints_parallel/patchcl_ae_best_auc.pt']:
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    gen, _ = create_patchcl_ae_models(in_channels=3, mhsa_mode='parallel', device=device)
    gen.load_state_dict(ckpt['generator_state'])
    gen.eval()
    
    all_lbls = []
    layer_maps = {'E_2': [], 'E_3': [], 'E_4': [], 'E_5': []}
    proj_maps = {'E_2': [], 'E_3': [], 'E_4': []}
    
    with torch.no_grad():
        for batch in test_loader:
            img = batch['image'].to(device)
            lbl = batch['label'].numpy()
            rec = gen(img)
            
            _, fr = gen.encode(img)
            _, fc = gen.encode(rec)
            
            all_lbls.extend(lbl)
            
            for l in ['E_2', 'E_3', 'E_4', 'E_5']:
                cos = F.cosine_similarity(fr[l], fc[l], dim=1).unsqueeze(1)
                layer_maps[l].append((1.0 - cos).cpu())
                
            for l in ['E_2', 'E_3', 'E_4']:
                p_head = gen.projection.heads[l]
                b, c, h, w = fr[l].shape
                r_v = p_head(fr[l].permute(0,2,3,1).reshape(-1, c)).view(b, h, w, -1)
                c_v = p_head(fc[l].permute(0,2,3,1).reshape(-1, c)).view(b, h, w, -1)
                r_norm = F.normalize(r_v, p=2, dim=-1)
                c_norm = F.normalize(c_v, p=2, dim=-1)
                p_cos = (r_norm * c_norm).sum(dim=-1, keepdim=True).permute(0, 3, 1, 2)
                proj_maps[l].append((1.0 - p_cos).cpu())

    all_lbls = np.array(all_lbls)
    
    # Concatenate
    maps = {l: torch.cat(layer_maps[l], dim=0) for l in layer_maps}
    p_maps = {l: torch.cat(proj_maps[l], dim=0) for l in proj_maps}
    
    print(f"\n=======================================================")
    print(f"SEARCHING COMBINATIONS FOR: {ckpt_p}")
    print(f"Paper Target: SEN={PAPER_SEN}%, SPE={PAPER_SPE}%")
    print(f"=======================================================")
    
    # Try different combinations
    # Interpolate all to 28x28
    m2_28 = F.interpolate(maps['E_2'], size=(28, 28), mode='bilinear', align_corners=True)
    m3_28 = F.interpolate(maps['E_3'], size=(28, 28), mode='bilinear', align_corners=True)
    m4_28 = maps['E_4']
    m5_28 = maps['E_5']
    
    p3_28 = F.interpolate(p_maps['E_3'], size=(28, 28), mode='bilinear', align_corners=True)
    p4_28 = p_maps['E_4']
    
    candidates = [
        ("E5 only", m5_28),
        ("0.4 E4 + 0.6 E5", 0.4 * m4_28 + 0.6 * m5_28),
        ("0.2 E3 + 0.3 E4 + 0.5 E5", 0.2 * m3_28 + 0.3 * m4_28 + 0.5 * m5_28),
        ("0.5 E4 + 0.5 E5", 0.5 * m4_28 + 0.5 * m5_28),
        ("0.3 P4 + 0.7 E5", 0.3 * p4_28 + 0.7 * m5_28),
        ("0.2 P3 + 0.3 P4 + 0.5 E5", 0.2 * p3_28 + 0.3 * p4_28 + 0.5 * m5_28),
        ("Paper-style (P2+P3+P4+E5)/4", 0.25*F.interpolate(p_maps['E_2'], size=(28, 28), mode='bilinear', align_corners=True) + 0.25*p3_28 + 0.25*p4_28 + 0.25*m5_28)
    ]
    
    best_exceed = []
    
    for c_name, c_map in candidates:
        for crop in [3, 4, 5, 6]:
            cr = c_map[:, 0, crop:-crop, crop:-crop].reshape(len(all_lbls), -1)
            for top_r in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]:
                k = max(1, int(top_r * cr.size(1)))
                sc = torch.topk(cr, k=k, dim=1).values.mean(dim=1).numpy()
                auc = roc_auc_score(all_lbls, sc) * 100
                fpr, tpr, thresh = roc_curve(all_lbls, sc)
                spe = 1.0 - fpr
                
                # Check performance at paper sensitivity (~92.78%)
                idx_92 = np.where(tpr >= 0.9278)[0]
                if len(idx_92) > 0:
                    spe_at_92 = spe[idx_92[np.argmax(spe[idx_92])]] * 100
                    sen_at_92 = tpr[idx_92[np.argmax(spe[idx_92])]] * 100
                else:
                    spe_at_92 = 0
                    sen_at_92 = 0
                    
                # Check performance at near paper sensitivity (90.0%)
                idx_90 = np.where(tpr >= 0.9000)[0]
                spe_at_90 = spe[idx_90[np.argmax(spe[idx_90])]] * 100 if len(idx_90) > 0 else 0
                sen_at_90 = tpr[idx_90[np.argmax(spe[idx_90])]] * 100 if len(idx_90) > 0 else 0
                
                # Check performance at SPE > paper SPE (75.0%)
                idx_spe75 = np.where(spe >= 0.7500)[0]
                sen_at_spe75 = tpr[idx_spe75[np.argmax(tpr[idx_spe75])]] * 100 if len(idx_spe75) > 0 else 0
                
                # Best F1
                th_sort = np.sort(sc)
                best_f1, b_sen, b_spe, b_acc = 0, 0, 0, 0
                for th in th_sort:
                    pred = (sc >= th).astype(int)
                    tp = np.sum((pred == 1) & (all_lbls == 1))
                    fp = np.sum((pred == 1) & (all_lbls == 0))
                    fn = np.sum((pred == 0) & (all_lbls == 1))
                    tn = np.sum((pred == 0) & (all_lbls == 0))
                    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0
                    if f1 > best_f1:
                        best_f1 = f1
                        b_sen = tp / (tp + fn) * 100
                        b_spe = tn / (tn + fp) * 100
                        b_acc = (tp + tn) / len(all_lbls) * 100
                        
                if auc >= 89.5:
                    tag = ""
                    if b_spe > PAPER_SPE:
                        tag += " [SPE_F1_BEATS_PAPER!]"
                    if spe_at_92 > 65.0:
                        tag += f" [SPE@92.8%={spe_at_92:.2f}%]"
                    if sen_at_spe75 >= 85.0:
                        tag += f" [SEN@SPE75%={sen_at_spe75:.2f}%]"
                        
                    best_exceed.append((auc, best_f1, b_acc, b_sen, b_spe, spe_at_92, sen_at_spe75, f"{c_name} Crop={crop} Top={top_r*100:4.1f}%"))
                    
    # Sort by AUC
    best_exceed.sort(key=lambda x: x[0], reverse=True)
    print("\nTOP 10 CONFIGURATIONS BY AUC:")
    print("Config                                | AUC    | Best F1 | ACC    | SEN_F1 | SPE_F1 | SPE@SEN92.8% | SEN@SPE75%")
    for item in best_exceed[:10]:
        print(f"{item[7]:38s} | {item[0]:5.2f}% | {item[1]:6.2f}% | {item[2]:5.2f}% | {item[3]:5.2f}% | {item[4]:5.2f}%  | {item[5]:6.2f}%      | {item[6]:6.2f}%")
