"""
Model loading utilities for FDKD pipeline.

IMPORTANT: Models MUST be created with MMPretrain's architecture
(ImageClassifier with ResNet/SwinTransformer backbone + GlobalAveragePooling
neck + LinearClsHead) because the checkpoints were trained with MMPretrain.
Using torchvision or timm models results in key mismatches and broken inference.
"""
from __future__ import annotations

import os
import torch

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


def _extract_state_dict(checkpoint_path: str, role: str = "") -> dict | None:
    """Load and extract state_dict from a checkpoint file.

    For distillation checkpoints (MMRazor), keys have the form:
        architecture.backbone.layer1...  (student)
        teacher.backbone.layer1...       (teacher)

    For standalone MMPretrain checkpoints, keys have the form:
        backbone.layer1...
        head.fc.weight

    This function:
    1. Loads the checkpoint
    2. Extracts the state_dict
    3. Filters by role prefix if needed (architecture. or teacher.)
    4. Returns the cleaned state_dict with original MMPretrain structure
       (backbone.*, head.*, neck.* keys preserved)
    """
    if not os.path.exists(checkpoint_path):
        print(f"  WARNING: File not found: {checkpoint_path}")
        return None

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

    sample_keys = list(state_dict.keys())[:8]
    print(f"  Raw keys (first 8): {sample_keys}")

    # Determine if this is a distillation checkpoint
    # MMRazor distillation: architecture.backbone.*, teacher.backbone.*
    # Standalone MMPretrain: backbone.*, head.*
    has_architecture = any(k.startswith("architecture.") for k in state_dict)
    has_teacher = any(k.startswith("teacher.") for k in state_dict)
    has_student = any(k.startswith("student.") for k in state_dict)
    is_distill = has_architecture or has_teacher or has_student

    if is_distill:
        # In MMRazor's SingleTeacherDistill:
        #   student/architecture → "architecture." prefix
        #   teacher → "teacher." prefix
        if role == "student":
            # Try "architecture." first (MMRazor convention), then "student."
            for prefix in ["architecture.", "student."]:
                filtered = {k[len(prefix):]: v for k, v in state_dict.items()
                            if k.startswith(prefix)}
                if filtered:
                    print(f"  Extracted {len(filtered)} keys with prefix '{prefix}'")
                    return filtered
        elif role == "teacher":
            prefix = "teacher."
            filtered = {k[len(prefix):]: v for k, v in state_dict.items()
                        if k.startswith(prefix)}
            if filtered:
                print(f"  Extracted {len(filtered)} keys with prefix '{prefix}'")
                return filtered
        else:
            # No role specified, try to use the full state_dict
            print(f"  ⚠️ Distillation checkpoint but no role specified")

    # Strip "module." prefix if present (DataParallel wrapper)
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = v

    return cleaned


def _create_mmpretrain_model(arch: str):
    """Create an MMPretrain ImageClassifier model.

    Args:
        arch: One of 'swin_base', 'resnet152', 'resnet18'

    Returns:
        An mmpretrain ImageClassifier model, or None if mmpretrain
        is not installed.

    Uses direct class instantiation instead of dict-based config
    to avoid mmengine registry overhead (dataclass hang on Colab).
    """
    try:
        from mmpretrain.models.classifiers import ImageClassifier
        from mmpretrain.models.backbones import ResNet, SwinTransformer
        from mmpretrain.models.necks import GlobalAveragePooling
        from mmpretrain.models.heads import LinearClsHead
        from torch.nn import CrossEntropyLoss

        if arch == "swin_base":
            backbone = SwinTransformer(arch='base', drop_path_rate=0.1, img_size=224)
            neck = GlobalAveragePooling()
            head = LinearClsHead(
                num_classes=NUM_CLASSES,
                in_channels=1024,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            )
        elif arch == "resnet152":
            backbone = ResNet(depth=152, num_stages=4, out_indices=(3,), style='pytorch')
            neck = GlobalAveragePooling()
            head = LinearClsHead(
                num_classes=NUM_CLASSES,
                in_channels=2048,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            )
        elif arch == "resnet18":
            backbone = ResNet(depth=18, num_stages=4, out_indices=(3,), style='pytorch')
            neck = GlobalAveragePooling()
            head = LinearClsHead(
                num_classes=NUM_CLASSES,
                in_channels=512,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Build ImageClassifier with pre-instantiated components
        model = ImageClassifier(
            backbone=backbone,
            neck=neck,
            head=head,
        )
        print(f"  Created MMPretrain model: {arch}")
        return model

    except ImportError as e:
        print(f"  ⚠️ mmpretrain not available ({e}), cannot create {arch}")
        return None
    except Exception as e:
        print(f"  ❌ Failed to create {arch}: {e}")
        return None


def _load_into_model(model, state_dict: dict, label: str) -> bool:
    """Load a state_dict into a model with diagnostics."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    matched = model_keys & ckpt_keys
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    print(f"  Keys: {len(matched)} matched, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")

    if len(matched) == 0:
        print(f"  ❌ No keys matched at all!")
        if missing:
            print(f"  Model expects (sample): {list(missing)[:3]}")
        if unexpected:
            print(f"  Checkpoint has (sample): {list(unexpected)[:3]}")
        return False

    match_ratio = len(matched) / len(model_keys)
    if match_ratio < 0.5:
        print(f"  ❌ Only {match_ratio:.0%} keys matched — refusing to load")
        return False

    try:
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"  ⚠️ Missing keys: {result.missing_keys[:5]}")
        print(f"  ✅ Loaded {label}: {len(matched)}/{len(model_keys)} keys "
              f"({match_ratio:.0%})")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def _try_load(model, candidate_dirs: list, model_name: str, role: str = "") -> bool:
    """Try loading checkpoint from a prioritized list of directories."""
    for dirname in candidate_dirs:
        dirpath = os.path.join(CHECKPOINT_DIR, dirname)
        if not os.path.exists(dirpath):
            dirpath = os.path.join(CHECKPOINT_DIR, dirname + ".pth")
        if os.path.exists(dirpath):
            print(f"\n[{model_name}] Trying: {dirname}")
            ckpt_path = find_best_checkpoint(dirpath)
            if ckpt_path:
                state_dict = _extract_state_dict(ckpt_path, role=role)
                if state_dict and _load_into_model(
                    model, state_dict, os.path.basename(ckpt_path)
                ):
                    print(f"[{model_name}] ✅ Loaded from {dirname}")
                    return True
    print(f"[{model_name}] ❌ No checkpoint loaded")
    return False


def _preprocess_checkpoint(src_path: str, role: str) -> str | None:
    """Pre-process a distillation checkpoint into a clean standalone file.

    If the checkpoint contains architecture.*/teacher.* prefixes (MMRazor
    distillation format), extract the relevant weights, strip the prefix,
    and save a *_clean.pth file next to the original.

    Returns the path to the clean file (or original if already clean).
    Skips processing if the clean file already exists (cached).
    """
    if not os.path.exists(src_path):
        return None

    # Determine clean file path
    base, ext = os.path.splitext(src_path)
    clean_path = base + "_clean" + ext

    # Cache: skip if already preprocessed
    if os.path.exists(clean_path):
        print(f"  ♻️  Using cached clean checkpoint: {os.path.basename(clean_path)}")
        return clean_path

    # Load and inspect
    checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        sd = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        sd = checkpoint

    has_arch = any(k.startswith("architecture.") for k in sd)
    has_teacher_keys = any(k.startswith("teacher.") for k in sd)
    is_distill = has_arch or has_teacher_keys

    if not is_distill:
        # Already a standalone checkpoint — no preprocessing needed
        return src_path

    # Extract the relevant subset based on role
    clean_sd = {}
    if role == "student":
        for prefix in ["architecture.", "student."]:
            clean_sd = {k[len(prefix):]: v for k, v in sd.items()
                        if k.startswith(prefix)}
            if clean_sd:
                print(f"  🔧 Extracted {len(clean_sd)} student keys "
                      f"(prefix '{prefix}') from {os.path.basename(src_path)}")
                break
    elif role == "teacher":
        prefix = "teacher."
        clean_sd = {k[len(prefix):]: v for k, v in sd.items()
                    if k.startswith(prefix)}
        if clean_sd:
            print(f"  🔧 Extracted {len(clean_sd)} teacher keys "
                  f"from {os.path.basename(src_path)}")

    if not clean_sd:
        print(f"  ⚠️ Could not extract {role} keys from "
              f"{os.path.basename(src_path)}")
        return src_path

    # Save clean checkpoint
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    torch.save({"state_dict": clean_sd, "meta": meta}, clean_path)
    size_mb = os.path.getsize(clean_path) / (1024 * 1024)
    print(f"  💾 Saved clean checkpoint: {os.path.basename(clean_path)} "
          f"({len(clean_sd)} keys, {size_mb:.1f} MB)")

    return clean_path


# Mapping: (candidate_dir_name, role_to_extract)
_PREPROCESS_MAP = {
    # Distillation checkpoints that need student extraction
    "swinb_r152":       "student",
    "dkd_swinb_r152":   "student",
    "swinb_r18":        "student",
    "dkd_swinb_r18":    "student",
    "disilledr152_r18": "student",
    "dkd_r152_r18":     "student",
}


def preprocess_checkpoints():
    """Auto-detect and preprocess distillation checkpoints.

    Scans CHECKPOINT_DIR for known distillation checkpoints, extracts
    student weights, and saves clean standalone files. Clean files are
    cached — this only runs once per checkpoint.
    """
    if not os.path.exists(CHECKPOINT_DIR):
        return

    print(f"\n{'─'*50}")
    print("🔧 Auto-preprocessing checkpoints...")
    print(f"{'─'*50}")

    processed = 0
    for name, role in _PREPROCESS_MAP.items():
        # Try both directory and .pth file
        for candidate in [
            os.path.join(CHECKPOINT_DIR, name),
            os.path.join(CHECKPOINT_DIR, name + ".pth"),
        ]:
            if os.path.exists(candidate):
                ckpt_path = find_best_checkpoint(candidate)
                if ckpt_path:
                    result = _preprocess_checkpoint(ckpt_path, role)
                    if result and result != ckpt_path:
                        processed += 1
                break

    if processed > 0:
        print(f"\n  ✅ Preprocessed {processed} checkpoint(s)")
    else:
        print(f"\n  ✅ All checkpoints already clean or cached")

    print(f"{'─'*50}\n")


def load_all_models() -> dict:
    """Load Teacher (Swin-B), Assistant (ResNet-152), Student (ResNet-18).

    Automatically preprocesses distillation checkpoints on first run,
    then uses MMPretrain's native model architecture to match the
    checkpoint structure exactly.
    """
    global MODELS

    # ── Step 0: Auto-preprocess distillation checkpoints ──
    preprocess_checkpoints()

    teacher = _create_mmpretrain_model("swin_base")
    assistant = _create_mmpretrain_model("resnet152")
    student = _create_mmpretrain_model("resnet18")

    entries = os.listdir(CHECKPOINT_DIR) if os.path.exists(CHECKPOINT_DIR) else []
    print(f"\nCheckpoint directory: {CHECKPOINT_DIR}")
    print(f"Entries: {entries}\n")

    if teacher is not None:
        _try_load(teacher, TEACHER_DIRS, "Teacher (Swin-B)", role="teacher")
    if assistant is not None:
        _try_load(assistant, ASSISTANT_DIRS, "Assistant (ResNet-152)", role="student")
    if student is not None:
        _try_load(student, STUDENT_DIRS, "Student (ResNet-18)", role="student")

    for name, model in [("teacher", teacher), ("assistant", assistant), ("student", student)]:
        if model is not None:
            model = model.to(DEVICE).eval()
            MODELS[name] = model
        else:
            MODELS[name] = None

    print(f"\n{'='*50}")
    print(f"Models loaded on {DEVICE}")
    for k, v in MODELS.items():
        status = "✅ loaded" if v is not None else "❌ missing"
        print(f"  {k}: {status}")
    print(f"{'='*50}\n")
    return MODELS
