"""
GradCAM-style visualization for FDKD models.
- Teacher (Swin-B): Attention map from 7x7 spatial features × FC weights
- ResNet-18 (DKD/TAKD/Baseline): Input gradient saliency
"""
from __future__ import annotations

import torch
import numpy as np

from backend.models.loader import MODELS
from backend.models.swin_pure import SwinBackbone
from utils.config import DEVICE, IMAGE_SIZE, MODEL_ROLES, CHECKPOINT_DIR
import torch.nn as nn


# ── Swin-B: FC-weight attention map ──────────────

def _swin_heatmap(input_tensor, class_idx):
    """
    For Swin-B: get 7x7 spatial features from last stage,
    weight by FC row for target class -> activation map.
    No gradient needed, works with the pure-pytorch swin_pure model.
    """
    # Load SwinBackbone from raw checkpoint (same keys as before, proven to work)
    import os
    ckpt_path = _find_teacher_ckpt()
    if ckpt_path is None:
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # Build backbone
    backbone = SwinBackbone()
    backbone_keys = [k for k in sd if k.startswith("backbone.")]
    mapped_backbone = {}
    for k in backbone_keys:
        mapped_backbone[k[len("backbone."):]] = sd[k]
    backbone.load_state_dict(mapped_backbone, strict=False)

    # Build head
    num_features = backbone.num_features
    fc = nn.Linear(num_features, 200)
    if "head.fc.weight" in sd:
        fc.weight.data.copy_(sd["head.fc.weight"])
        fc.bias.data.copy_(sd["head.fc.bias"])

    backbone.to(DEVICE).eval()
    fc.to(DEVICE).eval()

    x = input_tensor.clone().to(DEVICE)

    with torch.no_grad():
        # Forward through backbone
        x, hw = backbone.patch_embed(x)
        for stage in backbone.stages:
            x, hw = stage(x, hw)
        # x: [1, 49, C], hw: (7, 7)

        H, W = hw
        C = x.shape[-1]

        # Get FC weight for target class
        w = fc.weight[class_idx]  # [C]

        # Compute per-position contribution: x · w
        # x: [1, H*W, C] -> [1, H*W]
        activation = (x.squeeze(0) * w.unsqueeze(0)).sum(dim=-1)  # [49]

        # Add bias (constant offset, skip for visualization)

        # Reshape to 7x7
        activation = activation.view(H, W).cpu().numpy()

        # Normalize
        act_min, act_max = activation.min(), activation.max()
        if act_max - act_min > 1e-8:
            activation = (activation - act_min) / (act_max - act_min)

    return _smooth(activation)


def _find_teacher_ckpt():
    import os
    for name in ["swinb_fully.pth", "swinbase_fully.pth"]:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            return path
    swin_dir = os.path.join(CHECKPOINT_DIR, "swinb_fully")
    if os.path.isdir(swin_dir):
        pths = [f for f in os.listdir(swin_dir) if f.endswith(".pth")]
        if pths:
            return os.path.join(swin_dir, pths[0])
    return None


# ── ResNet: Gradient saliency ────────────────────

def _cnn_heatmap(model, input_tensor, class_idx):
    """Input gradient saliency for CNN models (works on traced)."""
    x = input_tensor.clone().to(DEVICE).requires_grad_(True)
    output = model(x)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    score = output[0, class_idx]
    score.backward()

    grad = x.grad
    if grad is None:
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5

    saliency = grad.abs().max(dim=1)[0].squeeze(0).detach().cpu().numpy()
    smin, smax = saliency.min(), saliency.max()
    if smax - smin > 1e-8:
        saliency = (saliency - smin) / (smax - smin)

    return _smooth(saliency)


def _smooth(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)


# ── Public API ──────────────────────────────────

def generate_gradcam(input_tensor, class_idx):
    results = {}
    for role in MODEL_ROLES:
        model = MODELS.get(role)
        if model is None:
            results[role] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            continue
        try:
            if role == "teacher":
                results[role] = _swin_heatmap(input_tensor, class_idx)
            else:
                results[role] = _cnn_heatmap(model, input_tensor, class_idx)
        except Exception as e:
            print(f"  [gradcam:{role}] Error: {e}")
            results[role] = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5
    return results
