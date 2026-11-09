import os
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from PatchCL_AE_MHSA.models import create_patchcl_ae_models
from PatchCL_AE_MHSA.dataset import create_patchcl_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaders = create_patchcl_dataloaders('datasets/chest_xray', batch_size=16, num_workers=0)
test_loader = loaders['test']

print("==========================================================================")
print("EVALUATION USING THE EXACT PAPER PROTOCOL (THRESHOLD AT BEST F1-SCORE)")
print("==========================================================================")

for ckpt_p in ['PatchCL_AE_MHSA/checkpoints_parallel/patchcl_ae_epoch_010.pt',
               'PatchCL_AE_MHSA/checkpoints_parallel/patchcl_ae_best_auc.pt',
               'PatchCL_AE_MHSA/checkpoints_parallel/patchcl_ae_latest.pt']:
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    ep = ckpt.get('epoch', 'N/A')
    gen, _ = create_patchcl_ae_models(in_channels=3, mhsa_mode='parallel', device=device)
    gen.load_state_dict(ckpt['generator_state'])
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
    
    # Paper method: Find threshold maximizing F1-score
    best_f1, best_th, best_sen, best_spe, best_acc = 0, 0, 0, 0, 0
    th_candidates = np.sort(scores)
    for th in th_candidates:
        pred = (scores >= th).astype(int)
        tp = np.sum((pred == 1) & (all_lbls == 1))
        fp = np.sum((pred == 1) & (all_lbls == 0))
        fn = np.sum((pred == 0) & (all_lbls == 1))
        tn = np.sum((pred == 0) & (all_lbls == 0))
        if 2*tp + fp + fn > 0:
            f1 = 2 * tp / (2 * tp + fp + fn) * 100
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_sen = tp / (tp + fn) * 100
                best_spe = tn / (tn + fp) * 100
                best_acc = (tp + tn) / len(all_lbls) * 100
                
    base_name = os.path.basename(ckpt_p)
    print(f"Checkpoint: {base_name:28s} (Epoch {ep})")
    print(f"  AUC:      {auc:.2f}%")
    print(f"  F1-score: {best_f1:.2f}%")
    print(f"  ACC:      {best_acc:.2f}%")
    print(f"  SEN:      {best_sen:.2f}%")
    print(f"  SPE:      {best_spe:.2f}%")
    print(f"  Threshold: {best_th:.5f}\n")
