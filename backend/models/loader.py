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


def _extract_role_keys(state_dict: dict, role: str = "") -> dict:
    """Extract keys for a specific role from a distillation checkpoint.

    MMPretrain distillation checkpoints use prefixes like:
      student.backbone.layer1... / student.head.fc...
      teacher.backbone.layer1... / teacher.head.fc...

    This function filters by role prefix, then strips all known prefixes
    to produce clean keys matching a standalone torchvision/timm model.
    """
    STRIP_PREFIXES = ["module.", "model.", "backbone.", "head."]
    ROLE_PREFIXES = ["student.", "teacher.", "assistant."]

    # If a role is specified, first filter keys belonging to that role
    if role:
        role_prefix = f"{role}."
        filtered = {k[len(role_prefix):]: v for k, v in state_dict.items()
                     if k.startswith(role_prefix)}
        if filtered:
            state_dict = filtered

    # Strip known structural prefixes
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k
        # Remove any remaining role prefixes (e.g. if no role filter was used)
        for rp in ROLE_PREFIXES:
            if new_key.startswith(rp):
                new_key = new_key[len(rp):]
        # Remove structural prefixes
        for prefix in STRIP_PREFIXES:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = v

    return cleaned


def load_state_dict_flexible(model, checkpoint_path: str, role: str = "") -> bool:
    """Load checkpoint with flexible key handling.

    Args:
        model: The PyTorch model to load weights into.
        checkpoint_path: Path to .pth checkpoint file.
        role: Optional role name ('student', 'teacher', 'assistant') to
              extract from a distillation checkpoint that contains all roles.
    """
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

    # Log raw key prefixes for debugging
    sample_keys = list(state_dict.keys())[:5]
    print(f"  Raw checkpoint keys (first 5): {sample_keys}")

    # Try with role extraction first, then without
    for try_role in ([role, ""] if role else [""]):
        cleaned = _extract_role_keys(state_dict, try_role)
        role_label = f"role='{try_role}'" if try_role else "no role filter"

        model_keys = set(model.state_dict().keys())
        matched = model_keys & set(cleaned.keys())

        try:
            model.load_state_dict(cleaned, strict=True)
            print(f"  ✅ Loaded strict ({role_label}): "
                  f"{len(matched)}/{len(model_keys)} keys — "
                  f"{os.path.basename(checkpoint_path)}")
            return True
        except RuntimeError:
            if len(matched) > len(model_keys) * 0.5:
                model.load_state_dict(cleaned, strict=False)
                print(f"  ⚠️ Loaded non-strict ({role_label}): "
                      f"{len(matched)}/{len(model_keys)} keys — "
                      f"{os.path.basename(checkpoint_path)}")
                return True
            else:
                print(f"  ⏭️ Skipping ({role_label}): only "
                      f"{len(matched)}/{len(model_keys)} keys matched")

    # Last resort: load original state_dict non-strict
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"  ⚠️ Loaded (original keys, non-strict): "
              f"{os.path.basename(checkpoint_path)}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return False


def _try_load(model, candidate_dirs: list, model_name: str, role: str = "") -> bool:
    """Try loading checkpoint from a prioritized list of directories or flat .pth files."""
    for dirname in candidate_dirs:
        dirpath = os.path.join(CHECKPOINT_DIR, dirname)
        if not os.path.exists(dirpath):
            dirpath = os.path.join(CHECKPOINT_DIR, dirname + ".pth")
        if os.path.exists(dirpath):
            print(f"\n[{model_name}] Trying: {dirname}")
            ckpt = find_best_checkpoint(dirpath)
            if ckpt:
                if load_state_dict_flexible(model, ckpt, role=role):
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
        _try_load(teacher, TEACHER_DIRS, "Teacher (Swin-B)", role="teacher")
    _try_load(assistant, ASSISTANT_DIRS, "Assistant (ResNet-152)", role="student")
    _try_load(student, STUDENT_DIRS, "Student (ResNet-18)", role="student")

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
