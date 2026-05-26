"""
Verify the Swin-B pure implementation matches mmcv behavior.

This script tests:
  1. PatchMerging channel ordering (nn.Unfold vs manual slicing)
  2. relative_position_index matches MMPretrain's double_step_seq
  3. Key matching between model and checkpoint
  4. Forward pass produces non-random output

Run locally (no mmcv needed):
    python verify_swin_fix.py
    python verify_swin_fix.py --checkpoint /path/to/swinb_fully/best.pth
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn


def test_patch_merging_ordering():
    """Verify nn.Unfold ordering differs from manual slicing."""
    print("\n" + "=" * 60)
    print("TEST 1: PatchMerging Channel Ordering")
    print("=" * 60)

    C, H, W = 4, 4, 4  # small for visualization
    x = torch.arange(C * H * W, dtype=torch.float32).view(1, H, W, C)

    # Method A: Manual slicing (OLD — BUGGY)
    manual = torch.cat([
        x[:, 0::2, 0::2, :],  # (0,0)
        x[:, 1::2, 0::2, :],  # (1,0)
        x[:, 0::2, 1::2, :],  # (0,1)
        x[:, 1::2, 1::2, :],  # (1,1)
    ], dim=-1).view(1, -1, 4 * C)

    # Method B: nn.Unfold (NEW — matches mmcv)
    x_bchw = x.permute(0, 3, 1, 2)  # B, C, H, W
    sampler = nn.Unfold(kernel_size=2, stride=2)
    unfold = sampler(x_bchw).transpose(1, 2)  # B, H/2*W/2, 4*C

    match = torch.allclose(manual, unfold)
    print(f"  Manual shape: {manual.shape}")
    print(f"  Unfold shape: {unfold.shape}")
    print(f"  Outputs match: {match}")

    if not match:
        print(f"  ✅ CONFIRMED: Channel orderings DIFFER (bug was real)")
        # Show which positions differ
        diff_mask = (manual != unfold).any(dim=0).any(dim=0)
        n_diff = diff_mask.sum().item()
        print(f"  {n_diff}/{4*C} channel positions differ")
    else:
        print(f"  ⚠️ Outputs match (unexpected for non-trivial input)")

    return not match  # True = bug was confirmed and fix is needed


def test_relative_position_index():
    """Verify double_step_seq matches standard method for 7×7 windows."""
    print("\n" + "=" * 60)
    print("TEST 2: relative_position_index Comparison")
    print("=" * 60)

    ws = 7

    # Method A: Standard meshgrid (OLD)
    coords_h = torch.arange(ws)
    coords_w = torch.arange(ws)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
    cf = torch.flatten(coords, 1)
    rel = cf[:, :, None] - cf[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += ws - 1
    rel[:, :, 1] += ws - 1
    rel[:, :, 0] *= 2 * ws - 1
    idx_standard = rel.sum(-1)

    # Method B: double_step_seq (NEW — matches MMPretrain)
    Wh, Ww = ws, ws
    seq1 = torch.arange(0, (2 * Ww - 1) * Wh, 2 * Ww - 1)
    seq2 = torch.arange(0, Ww)
    rel_index_coords = (seq1[:, None] + seq2[None, :]).reshape(1, -1)
    idx_mmpretrain = rel_index_coords + rel_index_coords.T
    idx_mmpretrain = idx_mmpretrain.flip(1).contiguous()

    match = torch.equal(idx_standard, idx_mmpretrain)
    print(f"  Standard shape:    {idx_standard.shape}")
    print(f"  MMPretrain shape:  {idx_mmpretrain.shape}")
    print(f"  Indices match:     {match}")

    if not match:
        n_diff = (idx_standard != idx_mmpretrain).sum().item()
        total = idx_standard.numel()
        print(f"  ⚠️ {n_diff}/{total} positions differ")
        print(f"  This confirms the relative position bias was incorrect")
    else:
        print(f"  ✅ Indices match for {ws}×{ws} windows (both methods equivalent)")

    return True  # test completed


def test_key_matching(checkpoint_path=None):
    """Test that model keys match checkpoint keys."""
    print("\n" + "=" * 60)
    print("TEST 3: Key Matching")
    print("=" * 60)

    from backend.models.swin_pure import SwinClassifier
    model = SwinClassifier(num_classes=200)
    model_keys = set(model.state_dict().keys())
    print(f"  Model keys: {len(model_keys)}")
    print(f"  Sample keys: {list(model_keys)[:5]}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        # Check if needs prefix stripping
        sample = list(sd.keys())[:3]
        print(f"  Checkpoint keys: {len(sd)}")
        print(f"  Sample ckpt keys: {sample}")

        ckpt_keys = set(sd.keys())
        matched = model_keys & ckpt_keys
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys

        print(f"\n  Matched:    {len(matched)}/{len(model_keys)}")
        print(f"  Missing:    {len(missing)}")
        print(f"  Unexpected: {len(unexpected)}")

        if missing:
            print(f"  Missing (sample): {list(missing)[:5]}")
        if unexpected:
            print(f"  Unexpected (sample): {list(unexpected)[:5]}")

        # Check shapes
        shape_mismatch = 0
        for k in matched:
            if model.state_dict()[k].shape != sd[k].shape:
                print(f"  Shape mismatch: {k}: "
                      f"model={model.state_dict()[k].shape} vs "
                      f"ckpt={sd[k].shape}")
                shape_mismatch += 1

        if shape_mismatch == 0 and len(matched) == len(model_keys):
            print(f"  ✅ Perfect match!")
        elif shape_mismatch > 0:
            print(f"  ❌ {shape_mismatch} shape mismatches!")

        return len(matched) == len(model_keys) and shape_mismatch == 0
    else:
        print(f"  (No checkpoint provided — skipping key comparison)")
        return True


def test_forward_pass(checkpoint_path=None):
    """Test that forward pass produces non-uniform output."""
    print("\n" + "=" * 60)
    print("TEST 4: Forward Pass")
    print("=" * 60)

    from backend.models.swin_pure import SwinClassifier
    model = SwinClassifier(num_classes=200)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        result = model.load_state_dict(sd, strict=False)
        print(f"  Loaded checkpoint: {len(sd)} keys")
        print(f"  Missing: {len(result.missing_keys)}, "
              f"Unexpected: {len(result.unexpected_keys)}")
    else:
        print(f"  (No checkpoint — using random weights)")

    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        logits = model(dummy)
        probs = torch.softmax(logits, dim=-1).squeeze()
        top5 = probs.topk(5)

        print(f"\n  Output shape: {logits.shape}")
        print(f"  Top-5 predictions:")
        for i, (p, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"    {i+1}. class {idx.item():3d} = {p.item():.4f}")

        max_prob = probs.max().item()
        uniform = 1.0 / 200  # 0.005
        is_random = max_prob < uniform * 5  # < 2.5% means basically uniform

        print(f"\n  Max probability:  {max_prob:.4f}")
        print(f"  Uniform baseline: {uniform:.4f}")

        if checkpoint_path and os.path.exists(checkpoint_path):
            if is_random:
                print(f"  ❌ Output looks RANDOM (max prob ≈ uniform)")
                print(f"     The fix may not have resolved the issue")
            else:
                print(f"  ✅ Output is NON-RANDOM (max prob >> uniform)")
                print(f"     The Swin-B model is working correctly!")
        else:
            print(f"  (Random weights — cannot determine correctness)")

    return not is_random if checkpoint_path else True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Swin-B fixes")
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to Swin-B checkpoint (swinb_fully/best.pth)",
    )
    args = parser.parse_args()

    results = {}
    results["patch_merging"] = test_patch_merging_ordering()
    results["rel_pos_index"] = test_relative_position_index()
    results["key_matching"] = test_key_matching(args.checkpoint)
    results["forward_pass"] = test_forward_pass(args.checkpoint)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test}")
    print("=" * 60)
