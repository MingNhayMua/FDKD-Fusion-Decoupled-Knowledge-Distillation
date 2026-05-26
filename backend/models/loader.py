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
    "teacher":  "teacher_traced.pt",
    "dkd":      "dkd_traced.pt",
    "takd":     "takd_traced.pt",
    "baseline": "baseline_traced.pt",
}


def _load_traced(role: str, filename: str) -> torch.jit.ScriptModule | None:
    """Load a single traced model."""
    import time
    path = os.path.join(CHECKPOINT_DIR, filename)
    size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
    
    if not os.path.exists(path):
        print(f"  [{role}] ❌ Not found: {path}")
        return None

    print(f"  [{role}] Loading {filename} ({size_mb:.1f} MB) from Drive...", flush=True)
    t0 = time.time()

    try:
        model = torch.jit.load(path, map_location=DEVICE)
        print(f"  [{role}]    torch.jit.load done in {time.time()-t0:.0f}s", flush=True)
        model.eval()

        # Quick sanity check
        print(f"  [{role}]    Running sanity check...", flush=True)
        t1 = time.time()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
            out = model(dummy)
            print(f"  [{role}]    Done in {time.time()-t1:.0f}s, output shape: {out.shape}", flush=True)

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
