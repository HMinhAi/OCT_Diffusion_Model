import torch, glob, os
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from PatchCL_AE_MHSA.models import create_patchcl_ae_models
from PatchCL_AE_MHSA.dataset import create_patchcl_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaders = create_patchcl_dataloaders('datasets/chest_xray', batch_size=16, num_workers=0)
test_loader = loaders['test']

gen, _ = create_patchcl_ae_models(in_channels=3, mhsa_mode='parallel', device=device)

ckpts = sorted(glob.glob('PatchCL_AE_MHSA/checkpoints_parallel/*.pt'))

print("=== CHECKPOINT EVOLUTION (E4+E5 METRIC, CROP 5, TOP 25%) ===")
for p in ckpts:
    ckpt = torch.load(p, map_location=device, weights_only=False)
    ep = ckpt.get('epoch', 'N/A')
    gen.load_state_dict(ckpt['generator_state'])
    gen.eval()
    
    all_lbls, r4_list, c4_list, r5_list, c5_list = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            img = batch['image'].to(device)
            lbl = batch['label'].numpy()
            rec = gen(img)
            _, fr = gen.encode(img)
            _, fc = gen.encode(rec)
            r4_list.append(fr['E_4'])
            c4_list.append(fc['E_4'])
            r5_list.append(fr['E_5'])
            c5_list.append(fc['E_5'])
            all_lbls.extend(lbl)
            
    r4 = torch.cat(r4_list, dim=0)
    c4 = torch.cat(c4_list, dim=0)
    m4 = (1.0 - F.cosine_similarity(r4, c4, dim=1).unsqueeze(1))
    r5 = torch.cat(r5_list, dim=0)
    c5 = torch.cat(c5_list, dim=0)
    m5 = (1.0 - F.cosine_similarity(r5, c5, dim=1).unsqueeze(1))
    
    comb = 0.4 * m4 + 0.6 * m5
    comb_cr = comb[:, 0, 5:-5, 5:-5].reshape(len(all_lbls), -1)
    k = int(0.25 * comb_cr.size(1))
    scores = torch.topk(comb_cr, k=k, dim=1).values.mean(dim=1).cpu().numpy()
    
    all_lbls = np.array(all_lbls)
    auc = roc_auc_score(all_lbls, scores) * 100
    fpr, tpr, thresh = roc_curve(all_lbls, scores)
    spe = 1.0 - fpr
    v85 = np.where(spe >= 0.85)[0]
    sen85 = tpr[v85[np.argmax(tpr[v85])]] * 100.0 if len(v85) > 0 else 0
    v93 = np.where(tpr >= 0.93)[0]
    spe93 = spe[v93[np.argmax(spe[v93])]] * 100.0 if len(v93) > 0 else 0
    
    fname = os.path.basename(p)
    print(f"Epoch {ep:2} ({fname:26s}) | AUC: {auc:5.2f}% | @SPE>=85%: SEN={sen85:5.2f}% | @SEN>=93%: SPE={spe93:5.2f}%")
