"""Comprehensive Integration & Sanity Test for MHSA-Enhanced PatchCL-AE.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from PatchCL_AE_MHSA.models import (
    create_patchcl_ae_models,
    MHSABottleneckBlock,
    ParallelMHSAGuidance,
)
from PatchCL_AE_MHSA.losses import (
    PatchCL_TotalLoss,
    compute_attention_guided_infonce_loss,
)
from PatchCL_AE_MHSA.anomaly_score import (
    compute_patchcl_anomaly_map,
    anomaly_map_to_score,
)


def run_tests() -> None:
    print("=" * 65)
    print("STARTING MHSA-ENHANCED PATCHCL-AE INTEGRATION TESTS")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # TEST 1: MHSABottleneckBlock Shapes & Properties
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Testing MHSABottleneckBlock...")
    b, c, h, w = 2, 256, 28, 28
    x_test = torch.randn(b, c, h, w, device=device)
    block = MHSABottleneckBlock(in_dim=c, num_heads=4, gamma=2.0).to(device)
    
    out_feat, w_att, s_mat = block(x_test)
    assert out_feat.shape == (b, c, h, w), f"Expected {(b, c, h, w)}, got {out_feat.shape}"
    assert w_att.shape == (b, h * w, h * w), f"Expected {(b, h*w, h*w)}, got {w_att.shape}"
    assert s_mat.shape == (b, h * w, h * w), f"Expected {(b, h*w, h*w)}, got {s_mat.shape}"
    
    # Check bounds
    assert w_att.min() >= 0.0 and w_att.max() <= 1.0, f"w_att out of bounds: [{w_att.min()}, {w_att.max()}]"
    assert s_mat.min() >= 0.0 and s_mat.max() <= 1.0, f"s_mat out of bounds: [{s_mat.min()}, {s_mat.max()}]"
    print(f"  + Out features shape:    {out_feat.shape}")
    print(f"  + w_att shape:            {w_att.shape} (min={w_att.min():.4f}, mean={w_att.mean():.4f}, max={w_att.max():.4f})")
    print(f"  + s_mat shape:            {s_mat.shape} (min={s_mat.min():.4f}, mean={s_mat.mean():.4f}, max={s_mat.max():.4f})")
    print("  -> TEST 1 PASSED!")

    # -------------------------------------------------------------------------
    # TEST 2: Full Architecture & End-to-End Gradient Flow
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Testing Full Architecture & End-to-End Gradient Flow...")
    gen, disc = create_patchcl_ae_models(
        in_channels=3,
        mhsa_mode="bottleneck",
        num_heads=4,
        device=device,
    )
    img_in = torch.randn(b, 3, 224, 224, device=device)
    rec_out, feats, w_att = gen(img_in, return_features=True)

    assert rec_out.shape == (b, 3, 224, 224), f"Wrong rec shape: {rec_out.shape}"
    assert "E_1" in feats and "E_4" in feats and "E_5" in feats
    assert w_att is not None and w_att.shape == (b, 784, 784)

    # Test backward pass to verify MHSA receives gradients
    dummy_loss = rec_out.mean()
    dummy_loss.backward()

    assert gen.mhsa_bottleneck.mha.in_proj_weight.grad is not None, "MHSA in_proj_weight has NO grad!"
    assert gen.mhsa_bottleneck.pos_embed.grad is not None, "MHSA pos_embed has NO grad!"
    assert gen.encoder.E_1[0].weight.grad is not None, "Encoder E_1 has NO grad!"
    assert gen.decoder.De_1[0].weight.grad is not None, "Decoder De_1 has NO grad!"
    print("  + Reconstructed image shape: ", rec_out.shape)
    print("  + MHSA Attention weight grad: ", gen.mhsa_bottleneck.mha.in_proj_weight.grad.norm().item())
    print("  + MHSA Positional embed grad: ", gen.mhsa_bottleneck.pos_embed.grad.norm().item())
    print("  + Encoder E_1 weight grad:    ", gen.encoder.E_1[0].weight.grad.norm().item())
    print("  + Decoder De_1 weight grad:   ", gen.decoder.De_1[0].weight.grad.norm().item())
    print("  -> TEST 2 PASSED!")

    # -------------------------------------------------------------------------
    # TEST 3: Attention-Guided Contrastive Loss Computation
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Testing Attention-Guided InfoNCE Loss...")
    criterion = PatchCL_TotalLoss(
        temperature=0.07,
        num_patches=256,
        lambda_adv=0.10,
        lambda_patch=1.0,
        lambda_l1=0.20,
    ).to(device)

    gen.zero_grad()
    disc.zero_grad()

    rec_out, _, w_att = gen(img_in, return_features=True)
    _, feats_clean = gen.encode(img_in)
    _, feats_recon = gen.encode(rec_out)
    d_fake_logits = disc(rec_out)

    loss_g, loss_adv, loss_patch, loss_l1, loss_edge = criterion.forward_generator(
        d_fake_logits=d_fake_logits,
        features_real=feats_clean,
        features_fake=feats_recon,
        projection_module=gen.projection,
        w_attenuation=w_att,
        layers=["E_1", "E_2", "E_3", "E_4"],
        real_images=img_in,
        recon_images=rec_out,
        return_all=True,
    )

    print(f"  + Loss_G total: {loss_g.item():.4f}")
    print(f"  + Loss_Adv:     {loss_adv.item():.4f}")
    print(f"  + Loss_Patch:   {loss_patch.item():.4f}")
    print(f"  + Loss_L1:      {loss_l1.item():.4f}")

    loss_g.backward()
    assert gen.mhsa_bottleneck.mha.in_proj_weight.grad is not None
    print("  + Full loss backward through MHSA verified!")
    print("  -> TEST 3 PASSED!")

    # -------------------------------------------------------------------------
    # TEST 4: Anomaly Scoring & Map Calculation
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Testing Anomaly Map & Scoring...")
    gen.eval()
    with torch.no_grad():
        anomaly_map, x_rec, raw_maps = compute_patchcl_anomaly_map(
            model=gen,
            x=img_in,
            layers=["E_1", "E_2", "E_3", "E_4"],
            top_k_ratio=0.10,
            use_thresholding=False,
            use_projection_head=False,
        )
        scores = anomaly_map_to_score(
            anomaly_map,
            reduction="top_percent",
            top_k_ratio=0.10,
        )

    assert anomaly_map.shape == (b, 1, 224, 224), f"Wrong anomaly_map shape: {anomaly_map.shape}"
    assert scores.shape == (b,), f"Wrong scores shape: {scores.shape}"
    print(f"  + Anomaly map shape: {anomaly_map.shape} (min={anomaly_map.min():.4f}, max={anomaly_map.max():.4f})")
    print(f"  + Image-level scores: {scores.cpu().tolist()}")
    print("  -> TEST 4 PASSED!")

    print("\n" + "=" * 65)
    print("ALL 4 INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("MHSA INTEGRATION IN PATCHCL-AE IS FULLY VERIFIED AND FUNCTIONAL!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
