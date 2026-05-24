"""
Tiny ImageNet class label utilities.
"""
import os
import json

from utils.config import NUM_CLASSES


_CLASS_LABELS: dict = {}


def load_labels(path: str = None):
    """Load Tiny ImageNet class labels from JSON file."""
    global _CLASS_LABELS

    candidates = [path] if path else []
    bundled = os.path.join(os.path.dirname(__file__), "tiny_imagenet_labels.json")
    candidates.append(bundled)

    for p in candidates:
        if p and os.path.exists(p):
            with open(p, "r") as f:
                _CLASS_LABELS = json.load(f)
            print(f"Loaded {len(_CLASS_LABELS)} class labels from {p}")
            return
    else:
        _CLASS_LABELS = {str(i): f"Class_{i}" for i in range(NUM_CLASSES)}
        print(f"Using generic labels (no label file found)")


def get_label(class_id: int) -> str:
    """Get human-readable class name by ID."""
    return _CLASS_LABELS.get(str(class_id), f"Class_{class_id}")
