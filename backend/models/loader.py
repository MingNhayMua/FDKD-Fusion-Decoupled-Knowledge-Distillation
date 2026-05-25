"""
Model loading utilities for FDKD pipeline.
"""
import os
import torch
from torchvision import models

from utils.config import (
    DEVICE, NUM_CLASSES, CHECKPOINT_DIR,
    TEACHER_DIRS, ASSISTANT_DIRS, STUDENT_DIRS,
    CHECKPOINT_EXTENSIONS,
)


# Global model registry
MODELS: dict = {}


def find_best_checkpoint(directory: str) -> str | None:
    """Find the best .pth checkpoint inside an MMPretrain output directory.

    Search priority: best_* > latest epoch > any checkpoint file.
    """
    if not os.path.isdir(directory):
        if directory.endswith(CHECKPOINT_EXTENSIONS):
            return directory
        return None

    all_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            all_files.append(os.path.join(root, f))

    rel_files = [os.path.relpath(f, directory) for f in all_files[:20]]
    print(f"  📂 Contents of {os.path.basename(directory)}/: {rel_files}")

    pth_files = [f for f in all_files if f.lower().endswith(CHECKPOINT_EXTENSIONS)]

    if not pth_files:
        print(f"  ❌ No checkpoint files found")
        return None

    print(f"  Found {len(pth_files)} checkpoints: "
          f"{[os.path.basename(f) for f in pth_files]}")

    # 1) Prefer 'best' checkpoint
    for pth in pth_files:
        if 'best' in os.path.basename(pth).lower():
            print(f"  ✅ Using best: {os.path.basename(pth)}")
            return pth

    # 2) Prefer highest epoch
    epoch_files = [p for p in pth_files
                   if 'epoch' in os.path.basename(p).lower()]
    if epoch_files:
        epoch_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        print(f"  ✅ Using latest epoch: {os.path.basename(epoch_files[0])}")
        return epoch_files[0]

    # 3) Fallback: most recent file
    pth_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    print(f"  ✅ Using: {os.path.basename(pth_files[0])}")
    return pth_files[0]


def load_state_dict_flexible(model, checkpoint_path: str) -> bool:
    """Load checkpoint with flexible key handling."""
    if not os.path.exists(checkpoint_path):
        print(f"  WARNING: File not found: {checkpoint_path}")
        return False

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip common prefixes (backbone., module., model.)
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in ["module.", "model.", "backbone."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = v

    try:
        model.load_state_dict(cleaned, strict=True)
        print(f"  Loaded (strict): {os.path.basename(checkpoint_path)}")
    except RuntimeError:
        try:
            model.load_state_dict(cleaned, strict=False)
            print(f"  Loaded (non-strict): {os.path.basename(checkpoint_path)}")
        except Exception:
            model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded (original keys): {os.path.basename(checkpoint_path)}")
    return True


def _try_load(model, candidate_dirs: list, model_name: str) -> bool:
    """Try loading checkpoint from a prioritized list of directories or flat .pth files."""
    for dirname in candidate_dirs:
        dirpath = os.path.join(CHECKPOINT_DIR, dirname)
        if not os.path.exists(dirpath):
            dirpath = os.path.join(CHECKPOINT_DIR, dirname + ".pth")
        if os.path.exists(dirpath):
            print(f"\n[{model_name}] Trying: {dirname}")
            ckpt = find_best_checkpoint(dirpath)
            if ckpt:
                if load_state_dict_flexible(model, ckpt):
                    print(f"[{model_name}] ✅ Loaded from {dirname}")
                    return True
    print(f"[{model_name}] ❌ No checkpoint loaded")
    return False


def load_all_models() -> dict:
    """Load Teacher (Swin-B), Assistant (ResNet-152), Student (ResNet-18)."""
    global MODELS

    try:
        import timm
        teacher = timm.create_model(
            "swin_base_patch4_window7_224",
            num_classes=NUM_CLASSES,
            pretrained=False,
        )
    except ImportError:
        print("⚠️ timm not installed — skipping Swin-B teacher")
        teacher = None

    assistant = models.resnet152(weights=None)
    assistant.fc = torch.nn.Linear(assistant.fc.in_features, NUM_CLASSES)

    student = models.resnet18(weights=None)
    student.fc = torch.nn.Linear(student.fc.in_features, NUM_CLASSES)

    entries = os.listdir(CHECKPOINT_DIR) if os.path.exists(CHECKPOINT_DIR) else []
    print(f"\nCheckpoint directory: {CHECKPOINT_DIR}")
    print(f"Entries: {entries}\n")

    if teacher is not None:
        _try_load(teacher, TEACHER_DIRS, "Teacher (Swin-B)")
    _try_load(assistant, ASSISTANT_DIRS, "Assistant (ResNet-152)")
    _try_load(student, STUDENT_DIRS, "Student (ResNet-18)")

    if teacher is not None:
        teacher = teacher.to(DEVICE).eval()
    assistant = assistant.to(DEVICE).eval()
    student = student.to(DEVICE).eval()

    MODELS["teacher"] = teacher
    MODELS["assistant"] = assistant
    MODELS["student"] = student

    print(f"\n{'='*50}")
    print(f"Models loaded on {DEVICE}")
    for k, v in MODELS.items():
        status = "✅ loaded" if v is not None else "❌ missing"
        print(f"  {k}: {status}")
    print(f"{'='*50}\n")
    return MODELS
