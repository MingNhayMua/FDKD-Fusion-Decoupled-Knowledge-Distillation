"""
Layer-by-layer debug: compare MMPretrain Swin vs pure Swin on Colab.

Run on Colab where mmcv/mmpretrain is available:
    %cd /content/FDKD-Fusion-Decoupled-Knowledge-Distillation
    %run debug_layerwise.py --checkpoint /content/drive/MyDrive/checkpoints/swinb_fully/best.pth
"""
import os, sys, argparse
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def stats(name, t):
    """Print tensor statistics."""
    if t.dim() > 1:
        print(f"  {name:40s} shape={str(list(t.shape)):20s} "
              f"mean={t.float().mean().item():+10.6f} "
              f"std={t.float().std().item():10.6f} "
              f"min={t.float().min().item():+10.6f} "
              f"max={t.float().max().item():+10.6f}")
    else:
        print(f"  {name:40s} shape={str(list(t.shape)):20s} "
              f"mean={t.float().mean().item():+10.6f}")


def run_mmpretrain(checkpoint_path, dummy_input):
    """Run MMPretrain Swin and capture intermediate outputs."""
    print("\n" + "="*70)
    print("MMPretrain Swin-B (ground truth)")
    print("="*70)

    try:
        from mmpretrain.models import (ImageClassifier, SwinTransformer,
                                        GlobalAveragePooling, LinearClsHead,
                                        CrossEntropyLoss)
    except ImportError:
        print("  ❌ mmpretrain not available — skipping")
        return None

    # Build model matching training config
    model = ImageClassifier(
        backbone=dict(
            type='SwinTransformer',
            arch='base',
            img_size=224,
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=200,
            in_channels=1024,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    result = model.load_state_dict(sd, strict=False)
    print(f"  Loaded: missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"  Missing: {result.missing_keys[:5]}")

    model.eval()
    backbone = model.backbone

    outputs = {}
    with torch.no_grad():
        # Patch embed
        x, hw = backbone.patch_embed(dummy_input)
        outputs['patch_embed'] = x.clone()
        stats("patch_embed", x)
        print(f"  hw_shape: {hw}")

        # Drop after pos (identity if drop_rate=0)
        x = backbone.drop_after_pos(x)

        # Stages
        for i, stage in enumerate(backbone.stages):
            # Run blocks only
            for j, block in enumerate(stage.blocks):
                x = block(x, hw)
                if j == 0:
                    outputs[f'stage{i}_block0'] = x.clone()
                    stats(f"stage{i}.block{j}", x)

            outputs[f'stage{i}_after_blocks'] = x.clone()
            stats(f"stage{i} after all blocks", x)

            # Downsample
            if stage.downsample is not None:
                x, hw = stage.downsample(x, hw)
                outputs[f'stage{i}_downsample'] = x.clone()
                stats(f"stage{i} after downsample", x)
                print(f"  hw_shape: {hw}")

        # Final norm
        norm = getattr(backbone, 'norm3')
        x = norm(x)
        outputs['final_norm'] = x.clone()
        stats("final_norm (norm3)", x)

        # GAP
        x_gap = x.mean(dim=1)
        outputs['gap'] = x_gap.clone()
        stats("GAP", x_gap)

        # Head
        logits = model.head.fc(x_gap)
        outputs['logits'] = logits.clone()
        probs = torch.softmax(logits, dim=-1).squeeze()
        top5 = probs.topk(5)
        print(f"\n  Top-5:")
        for k, (p, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"    {k+1}. class {idx.item():3d} = {p.item():.4f}")

    return outputs


def run_pure(checkpoint_path, dummy_input):
    """Run pure Swin and capture intermediate outputs."""
    print("\n" + "="*70)
    print("Pure PyTorch Swin-B")
    print("="*70)

    from backend.models.swin_pure import SwinClassifier
    model = SwinClassifier(num_classes=200)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    result = model.load_state_dict(sd, strict=False)
    print(f"  Loaded: missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"  Missing: {result.missing_keys[:5]}")

    model.eval()
    backbone = model.backbone

    outputs = {}
    with torch.no_grad():
        # Patch embed
        x, hw = backbone.patch_embed(dummy_input)
        outputs['patch_embed'] = x.clone()
        stats("patch_embed", x)
        print(f"  hw_shape: {hw}")

        # Stages
        for i, stage in enumerate(backbone.stages):
            for j, block in enumerate(stage.blocks):
                x = block(x, hw)
                if j == 0:
                    outputs[f'stage{i}_block0'] = x.clone()
                    stats(f"stage{i}.block{j}", x)

            outputs[f'stage{i}_after_blocks'] = x.clone()
            stats(f"stage{i} after all blocks", x)

            if stage.downsample is not None:
                x, hw = stage.downsample(x, hw)
                outputs[f'stage{i}_downsample'] = x.clone()
                stats(f"stage{i} after downsample", x)
                print(f"  hw_shape: {hw}")

        # Final norm
        x = backbone.norm3(x)
        outputs['final_norm'] = x.clone()
        stats("final_norm (norm3)", x)

        # GAP
        x_gap = x.mean(dim=1)
        outputs['gap'] = x_gap.clone()
        stats("GAP", x_gap)

        # Head
        logits = model.head.fc(x_gap)
        outputs['logits'] = logits.clone()
        probs = torch.softmax(logits, dim=-1).squeeze()
        top5 = probs.topk(5)
        print(f"\n  Top-5:")
        for k, (p, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"    {k+1}. class {idx.item():3d} = {p.item():.4f}")

    return outputs


def compare(mm_out, pure_out):
    """Compare outputs between MMPretrain and pure."""
    if mm_out is None:
        print("\n  (MMPretrain not available — cannot compare)")
        return

    print("\n" + "="*70)
    print("COMPARISON: MMPretrain vs Pure")
    print("="*70)

    for key in mm_out:
        if key in pure_out:
            a, b = mm_out[key], pure_out[key]
            if a.shape != b.shape:
                print(f"  {key:40s} SHAPE MISMATCH: {a.shape} vs {b.shape}")
                continue
            diff = (a - b).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            is_close = torch.allclose(a, b, atol=1e-5)
            status = "✅" if is_close else "❌"
            print(f"  {status} {key:38s} max_diff={max_diff:.8f} "
                  f"mean_diff={mean_diff:.8f}")

            if not is_close and key == 'patch_embed':
                print(f"     ⚠️ DIVERGENCE STARTS AT PATCH EMBED!")
            elif not is_close:
                # Find first divergence
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    dummy = torch.randn(1, 3, 224, 224)

    mm_out = run_mmpretrain(args.checkpoint, dummy.clone())
    pure_out = run_pure(args.checkpoint, dummy.clone())
    compare(mm_out, pure_out)
