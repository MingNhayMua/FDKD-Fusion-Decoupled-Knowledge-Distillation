"""
Debug script: compare MMPretrain Swin checkpoint keys vs timm Swin model keys.
Run this on Colab after loading the model.

Usage on Colab:
    %cd /content/FDKD-Fusion-Decoupled-Knowledge-Distillation
    %run debug_swin_keys.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_DIR = os.environ.get(
    "FDKD_CHECKPOINT_DIR",
    "/content/drive/MyDrive/CS338-checkpoint"
)

# 1. Load checkpoint
ckpt_path = os.path.join(CHECKPOINT_DIR, "swinb_fully.pth")
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]

# 2. Create timm model
import timm
model = timm.create_model("swin_base_patch4_window7_224", num_classes=200, pretrained=False)
model_sd = model.state_dict()

# 3. Apply key mapping
from backend.models.loader import _map_swin_keys
mapped_sd = _map_swin_keys(sd)

# 4. Compare
model_keys = set(model_sd.keys())
mapped_keys = set(mapped_sd.keys())
matched = model_keys & mapped_keys
missing = model_keys - mapped_keys
unexpected = mapped_keys - model_keys

print(f"\n{'='*60}")
print(f"Model keys:      {len(model_keys)}")
print(f"Mapped ckpt keys: {len(mapped_keys)}")
print(f"Matched:         {len(matched)}")
print(f"Missing:         {len(missing)}")
print(f"Unexpected:      {len(unexpected)}")

# 5. Check shape mismatches in matched keys
print(f"\n{'='*60}")
print("Shape mismatches in matched keys:")
shape_mismatches = 0
for k in sorted(matched):
    model_shape = model_sd[k].shape
    ckpt_shape = mapped_sd[k].shape
    if model_shape != ckpt_shape:
        print(f"  {k}: model={model_shape}, ckpt={ckpt_shape}")
        shape_mismatches += 1
if shape_mismatches == 0:
    print("  ✅ None! All shapes match perfectly.")
else:
    print(f"  ❌ {shape_mismatches} shape mismatches!")

# 6. List missing keys
if missing:
    print(f"\n{'='*60}")
    print("Missing keys (model has, checkpoint doesn't):")
    for k in sorted(missing):
        print(f"  {k}: {model_sd[k].shape}")

# 7. List unexpected keys  
if unexpected:
    print(f"\n{'='*60}")
    print("Unexpected keys (checkpoint has, model doesn't):")
    for k in sorted(unexpected):
        print(f"  {k}: {mapped_sd[k].shape}")

# 8. Test inference
print(f"\n{'='*60}")
print("Testing inference...")
result = model.load_state_dict(mapped_sd, strict=False)
print(f"  Missing: {len(result.missing_keys)}")
print(f"  Unexpected: {len(result.unexpected_keys)}")

model.eval()
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(dummy)
    probs = torch.softmax(logits, dim=-1).squeeze()
    top5 = probs.topk(5)
    print(f"\n  Top-5 predictions on random input:")
    for i, (p, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"    {i+1}. class {idx.item():3d} = {p.item():.4f}")
    print(f"  Max prob: {probs.max():.4f}")
    print(f"  Near uniform: {(probs > 0.003).sum().item()}/200")
