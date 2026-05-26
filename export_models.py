"""
FDKD — Export MMPretrain models to TorchScript traced format.

Run on Colab where mmcv/mmpretrain is available:
    %cd /content/FDKD-Fusion-Decoupled-Knowledge-Distillation
    %run export_models.py

This traces all 3 models (Teacher, Assistant, Student) and saves them as
standalone .pt files that can be loaded anywhere with torch.jit.load()
— no mmcv/mmpretrain/timm needed at inference time.

Output files (saved to CHECKPOINT_DIR):
    teacher_traced.pt   — Swin-B  (input: [B,3,224,224] → output: [B,200])
    assistant_traced.pt — ResNet-152
    student_traced.pt   — ResNet-18
"""
import os
import sys
import argparse

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TracedClassifier(nn.Module):
    """Wrapper that takes backbone output and applies GAP + FC.

    MMPretrain's backbone returns a tuple of feature maps.
    This wrapper extracts the last feature map, applies global average
    pooling, and passes through the classifier head.
    """

    def __init__(self, backbone, fc, is_cnn=False):
        super().__init__()
        self.backbone = backbone
        self.fc = fc
        self.is_cnn = is_cnn

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            x = feats[-1]
        else:
            x = feats
        # Global average pooling
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])  # B, C, H, W → B, C
        elif x.dim() == 3:
            x = x.mean(dim=1)  # B, L, C → B, C
        return self.fc(x)


def export_teacher(checkpoint_dir, output_dir):
    """Export Swin-B teacher model."""
    print("\n" + "=" * 60)
    print("Exporting Teacher (Swin-B)")
    print("=" * 60)

    from mmpretrain.models import (ImageClassifier, SwinTransformer,
                                    GlobalAveragePooling, LinearClsHead,
                                    CrossEntropyLoss)

    model = ImageClassifier(
        backbone=dict(type='SwinTransformer', arch='base', img_size=224),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead', num_classes=200, in_channels=1024,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    )

    # Find checkpoint
    ckpt_path = None
    for name in ['swinb_fully', 'swinbase_fully']:
        for candidate in [
            os.path.join(checkpoint_dir, name, 'best.pth'),
            os.path.join(checkpoint_dir, name, 'epoch_79.pth'),
            os.path.join(checkpoint_dir, name + '.pth'),
        ]:
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
        if ckpt_path:
            break

    if not ckpt_path:
        # Try any .pth in swinb_fully/
        swin_dir = os.path.join(checkpoint_dir, 'swinb_fully')
        if os.path.isdir(swin_dir):
            pths = [f for f in os.listdir(swin_dir) if f.endswith('.pth')]
            if pths:
                pths.sort(key=lambda f: os.path.getmtime(
                    os.path.join(swin_dir, f)), reverse=True)
                ckpt_path = os.path.join(swin_dir, pths[0])

    if not ckpt_path:
        print("  ❌ No Swin-B checkpoint found!")
        return False

    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    result = model.load_state_dict(sd, strict=False)
    print(f"  Missing: {len(result.missing_keys)}, "
          f"Unexpected: {len(result.unexpected_keys)}")
    model.eval()

    wrapper = TracedClassifier(model.backbone, model.head.fc)
    wrapper.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        # Verify before tracing
        out = wrapper(dummy)
        probs = torch.softmax(out, dim=-1)
        print(f"  Pre-trace max prob: {probs.max().item():.4f}")

        traced = torch.jit.trace(wrapper, dummy)

        # Verify after tracing
        out2 = traced(dummy)
        diff = (out - out2).abs().max().item()
        print(f"  Post-trace diff: {diff:.8f}")

    out_path = os.path.join(output_dir, 'teacher_traced.pt')
    torch.jit.save(traced, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✅ Saved: {out_path} ({size_mb:.1f} MB)")
    return True


def export_resnet(checkpoint_dir, output_dir, depth, role, dirs):
    """Export ResNet model (assistant or student)."""
    label = f"{'Assistant' if role == 'assistant' else 'Student'} (ResNet-{depth})"
    print(f"\n{'=' * 60}")
    print(f"Exporting {label}")
    print("=" * 60)

    from torchvision import models as tv_models

    if depth == 152:
        model = tv_models.resnet152(weights=None)
    else:
        model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)

    # Find checkpoint
    ckpt_path = None
    for dirname in dirs:
        for candidate in [
            os.path.join(checkpoint_dir, dirname, 'best.pth'),
            os.path.join(checkpoint_dir, dirname + '.pth'),
            os.path.join(checkpoint_dir, dirname + '_clean.pth'),
        ]:
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
        if not ckpt_path:
            d = os.path.join(checkpoint_dir, dirname)
            if os.path.isdir(d):
                pths = [f for f in os.listdir(d) if f.endswith('.pth')]
                if pths:
                    # Prefer _clean, then best, then latest
                    for pth in sorted(pths):
                        if '_clean' in pth:
                            ckpt_path = os.path.join(d, pth)
                            break
                    if not ckpt_path:
                        for pth in sorted(pths):
                            if 'best' in pth:
                                ckpt_path = os.path.join(d, pth)
                                break
                    if not ckpt_path:
                        ckpt_path = os.path.join(d, pths[-1])
        if ckpt_path:
            break

    if not ckpt_path:
        print(f"  ❌ No checkpoint found for {label}!")
        return False

    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)

    # Strip MMPretrain prefixes
    mapped = {}
    for k, v in sd.items():
        new_key = k
        if new_key.startswith('backbone.'):
            new_key = new_key[len('backbone.'):]
        elif new_key.startswith('head.'):
            new_key = new_key[len('head.'):]
        mapped[new_key] = v

    result = model.load_state_dict(mapped, strict=False)
    print(f"  Missing: {len(result.missing_keys)}, "
          f"Unexpected: {len(result.unexpected_keys)}")
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
        probs = torch.softmax(out, dim=-1)
        print(f"  Pre-trace max prob: {probs.max().item():.4f}")

        traced = torch.jit.trace(model, dummy)
        out2 = traced(dummy)
        diff = (out - out2).abs().max().item()
        print(f"  Post-trace diff: {diff:.8f}")

    filename = f"{'assistant' if role == 'assistant' else 'student'}_traced.pt"
    out_path = os.path.join(output_dir, filename)
    torch.jit.save(traced, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✅ Saved: {out_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export FDKD models to TorchScript")
    parser.add_argument("--checkpoint-dir",
                        default="/content/drive/MyDrive/checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (defaults to checkpoint-dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Output dir:     {output_dir}")

    from utils.config import TEACHER_DIRS, ASSISTANT_DIRS, STUDENT_DIRS

    results = {}
    results['teacher'] = export_teacher(args.checkpoint_dir, output_dir)
    results['assistant'] = export_resnet(
        args.checkpoint_dir, output_dir, 152, 'assistant', ASSISTANT_DIRS)
    results['student'] = export_resnet(
        args.checkpoint_dir, output_dir, 18, 'student', STUDENT_DIRS)

    print(f"\n{'=' * 60}")
    print("EXPORT SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    traced_files = [f for f in os.listdir(output_dir) if f.endswith('_traced.pt')]
    print(f"\nTraced files in {output_dir}:")
    for f in sorted(traced_files):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"  {f} ({size:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
