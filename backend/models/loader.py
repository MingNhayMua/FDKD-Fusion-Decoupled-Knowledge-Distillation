"""
Model loading utilities for FDKD pipeline.

Loads TorchScript traced models exported by export_models.py.
These are standalone .pt files — no mmcv/mmpretrain/timm needed.

Expected files in CHECKPOINT_DIR:
    teacher_traced.pt   — Swin-B  (input: [B,3,224,224] → output: [B,200])
    assistant_traced.pt — ResNet-152
    student_traced.pt   — ResNet-18
"""
from __future__ import annotations

import os
import torch

from utils.config import DEVICE, CHECKPOINT_DIR


# Global model registry
MODELS: dict = {}

# Traced model filenames
TRACED_FILES = {
    "teacher":   "teacher_traced.pt",
    "assistant": "assistant_traced.pt",
    "student":   "student_traced.pt",
}


def _load_traced(role: str, filename: str) -> torch.jit.ScriptModule | None:
    """Load a single traced model."""
    path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        print(f"  [{role}] ❌ Not found: {path}")
        return None

    try:
        model = torch.jit.load(path, map_location=DEVICE)
        model.eval()

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{role}] ✅ Loaded {filename} ({size_mb:.1f} MB)")

        # Quick sanity check
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
            out = model(dummy)
            print(f"  [{role}]    Output shape: {out.shape}")

        return model
    except Exception as e:
        print(f"  [{role}] ❌ Failed to load: {e}")
        return None


def load_all_models() -> dict:
    """Load all traced models.

    Models must first be exported using export_models.py on Colab.
    """
    global MODELS

    print(f"\n{'='*50}")
    print(f"Loading traced models from: {CHECKPOINT_DIR}")
    print(f"{'='*50}")

    if not os.path.exists(CHECKPOINT_DIR):
        print(f"  ❌ Checkpoint directory not found!")
        print(f"  Run export_models.py on Colab first.")
        MODELS = {"teacher": None, "assistant": None, "student": None}
        return MODELS

    # List available files
    traced_files = [f for f in os.listdir(CHECKPOINT_DIR)
                    if f.endswith('_traced.pt')]
    print(f"  Found traced files: {traced_files}\n")

    for role, filename in TRACED_FILES.items():
        MODELS[role] = _load_traced(role, filename)

    print(f"\n{'='*50}")
    print(f"Models loaded on {DEVICE}")
    for k, v in MODELS.items():
        status = "✅ loaded" if v is not None else "❌ missing"
        print(f"  {k}: {status}")
    print(f"{'='*50}\n")

    return MODELS
