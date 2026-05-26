"""
GradCAM-style visualization for all 4 FDKD models.
All use FC-weight activation: spatial features × FC weights = heatmap.
- Teacher (Swin-B): 7×7 spatial features from SwinBackbone
- ResNet-18 (DKD/TAKD/Baseline): 7×7 features from layer4 output
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models as tv_models

from utils.config import DEVICE, IMAGE_SIZE, MODEL_ROLES, CHECKPOINT_DIR


# ── Swin-B: FC-weight on 7×7 spatial features ────

def _swin_heatmap(input_tensor, class_idx):
    from backend.models.swin_pure import SwinBackbone

    ckpt_path = _find_teacher_ckpt()
    if ckpt_path is None:
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    backbone = SwinBackbone()
    mapped = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    backbone.load_state_dict(mapped, strict=False)

    fc = nn.Linear(backbone.num_features, 200)
    if "head.fc.weight" in sd:
        fc.weight.data.copy_(sd["head.fc.weight"])
        fc.bias.data.copy_(sd["head.fc.bias"])

    backbone.to(DEVICE).eval()
    fc.to(DEVICE).eval()

    x = input_tensor.clone().to(DEVICE)
    with torch.no_grad():
        x, hw = backbone.patch_embed(x)
        for stage in backbone.stages:
            x, hw = stage(x, hw)
        H, W = hw
        w = fc.weight[class_idx]
        activation = (x.squeeze(0) * w.unsqueeze(0)).sum(dim=-1).view(H, W).cpu().numpy()

    return _normalize_and_smooth(activation)


def _find_teacher_ckpt():
    for name in ["swinb_fully.pth", "swinbase_fully.pth"]:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            return path
    return None


# ── ResNet: FC-weight on layer4 7×7 features ─────

def _resnet_heatmap(role, input_tensor, class_idx):
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)

    ckpt_path = _find_resnet_ckpt(role)
    if ckpt_path is None:
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    mapped = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            mapped[k[len("backbone."):]] = v
        elif k.startswith("head."):
            mapped[k[len("head."):]] = v
        else:
            mapped[k] = v
    model.load_state_dict(mapped, strict=False)
    model.to(DEVICE).eval()

    x = input_tensor.clone().to(DEVICE)
    with torch.no_grad():
        # Forward up to layer4 (get final conv features)
        features = x
        features = model.conv1(features)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)  # [1, 512, 7, 7]

        B, C, H, W = features.shape
        features_flat = features.view(B, C, H * W)  # [1, 512, 49]

        w = model.fc.weight[class_idx]  # [512]

        # Per-position contribution: w · f[:, i]
        activation = (features_flat.squeeze(0).T * w.unsqueeze(0)).sum(dim=-1)  # [49]
        activation = activation.view(H, W).cpu().numpy()

    return _normalize_and_smooth(activation)


def _find_resnet_ckpt(role):
    ckpt_map = {
        "dkd": ["swinb_r18_clean.pth", "dkd_swinb_r18_clean.pth",
                "swinb_r18.pth", "dkd_swinb_r18.pth"],
        "takd": ["disilledr152_r18_clean.pth", "dkd_r152_r18_clean.pth",
                 "disilledr152_r18.pth", "dkd_r152_r18.pth"],
        "baseline": ["r18_fully.pth"],
    }
    for name in ckpt_map.get(role, []):
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            return path
    return None


# ── Helpers ──────────────────────────────────────

def _normalize_and_smooth(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    amin, amax = arr.min(), arr.max()
    if amax - amin > 1e-8:
        arr = (arr - amin) / (amax - amin)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)


# ── Public API ──────────────────────────────────

def generate_gradcam(input_tensor, class_idx):
    results = {}
    for role in MODEL_ROLES:
        try:
            if role == "teacher":
                results[role] = _swin_heatmap(input_tensor, class_idx)
            else:
                results[role] = _resnet_heatmap(role, input_tensor, class_idx)
        except Exception as e:
            print(f"  [gradcam:{role}] Error: {e}")
            results[role] = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5
    return results
