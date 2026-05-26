"""
GradCAM visualization for all 4 FDKD models.

- ResNet-18 (DKD, TAKD, Baseline): Standard GradCAM on layer4
- Swin-B (Teacher): GradCAM on final 7x7 spatial features before pooling
"""
from __future__ import annotations

import os
import io
import base64
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models as tv_models

from utils.config import CHECKPOINT_DIR, DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# ── ResNet GradCAM ──────────────────────────────────

class _ResNetGradCAM:
    """GradCAM for ResNet-18 using gradient of target class w.r.t. layer4 output."""

    def __init__(self, role: str):
        self.role = role
        self.model: Optional[nn.Module] = None
        self.target_activations: Optional[torch.Tensor] = None
        self.target_gradients: Optional[torch.Tensor] = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch

        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 200)

        filename = f"{self.role}_traced.pt"
        # Load from traced file — but we need raw weights for GradCAM
        # Load raw checkpoint instead
        ckpt_map = {
            "dkd": ["swinb_r18_clean.pth", "dkd_swinb_r18_clean.pth",
                    "swinb_r18.pth", "dkd_swinb_r18.pth"],
            "takd": ["disilledr152_r18_clean.pth", "dkd_r152_r18_clean.pth",
                     "disilledr152_r18.pth", "dkd_r152_r18.pth"],
            "baseline": ["r18_fully.pth"],
        }
        candidates = ckpt_map.get(self.role, [])
        ckpt_path = None
        for name in candidates:
            path = os.path.join(CHECKPOINT_DIR, name)
            if os.path.exists(path):
                ckpt_path = path
                break

        if not ckpt_path:
            print(f"  [gradcam:{self.role}] No raw checkpoint found")
            return

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
        model.to(DEVICE)
        model.eval()
        self.model = model

        # Register hooks on layer4
        def forward_hook(module, input, output):
            self.target_activations = output

        def backward_hook(module, grad_input, grad_output):
            self.target_gradients = grad_output[0]

        self.model.layer4.register_forward_hook(forward_hook)
        self.model.layer4.register_full_backward_hook(backward_hook)
        self._loaded = True
        print(f"  [gradcam:{self.role}] Loaded from {os.path.basename(ckpt_path)}")

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self._load()
        if self.model is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        self.target_activations = None
        self.target_gradients = None

        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        score = output[0, class_idx]

        self.model.zero_grad()
        score.backward(retain_graph=False)

        activations = self.target_activations  # [B, C, H, W]
        gradients = self.target_gradients      # [B, C, H, W]

        if activations is None or gradients is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # Global average pool gradients
        weights = gradients.mean(dim=[2, 3])  # [1, C]
        cam = (weights[:, :, None, None] * activations).sum(dim=1)  # [1, H, W]
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Swin-B Spatial Heatmap ──────────────────────────

def _load_swin_checkpoint():
    """Load Swin-B teacher from raw checkpoint using swin_pure.py."""
    from backend.models.swin_pure import SwinBackbone, PatchEmbed

    ckpt_path = None
    for name in ["swinb_fully.pth", "swinbase_fully.pth"]:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            ckpt_path = path
            break
    if not ckpt_path:
        swin_dir = os.path.join(CHECKPOINT_DIR, "swinb_fully")
        if os.path.isdir(swin_dir):
            pths = [f for f in os.listdir(swin_dir) if f.endswith(".pth")]
            if pths:
                ckpt_path = os.path.join(swin_dir, pths[0])

    if not ckpt_path:
        return None, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    backbone = SwinBackbone()
    backbone_keys = [k for k in sd if k.startswith("backbone.")]
    mapped = {}
    for k in backbone_keys:
        mapped[k[len("backbone."):]] = sd[k]
    backbone.load_state_dict(mapped, strict=False)

    head_module = nn.Module()
    head_module.fc = nn.Linear(backbone.num_features, 200)
    head_keys = [k for k in sd if k.startswith("head.")]
    for k in head_keys:
        if "fc" in k:
            new_k = k[len("head."):]
            head_module.state_dict()[new_k].copy_(sd[k])

    # Try direct load of head.fc
    if "head.fc.weight" in sd:
        head_module.fc.weight.data.copy_(sd["head.fc.weight"])
        head_module.fc.bias.data.copy_(sd["head.fc.bias"])

    backbone.to(DEVICE)
    head_module.to(DEVICE)
    backbone.eval()
    head_module.eval()

    return backbone, head_module


_swin_cached = None

def _get_swin():
    global _swin_cached
    if _swin_cached is None:
        _swin_cached = _load_swin_checkpoint()
    return _swin_cached


class _SwinGradCAM:
    """GradCAM on Swin-B final 7x7 spatial features (before GAP)."""

    def __init__(self):
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self.backbone, self.head = _get_swin()
        if self.backbone is not None:
            self._loaded = True

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self._load()
        backbone = self.backbone
        head = self.head

        if backbone is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad = True

        # Forward
        x, hw = backbone.patch_embed(input_tensor)
        for stage in backbone.stages:
            x, hw = stage(x, hw)

        H, W = hw  # 7x7 for 224x224 input
        C = x.shape[-1]
        spatial_feats = x.view(1, H, W, C).permute(0, 3, 1, 2)  # [1, C, H, W]

        pooled = backbone.norm3(x).mean(dim=1)
        logits = head.fc(pooled)
        score = logits[0, class_idx]

        backbone.zero_grad()
        head.zero_grad()
        score.backward()

        # Grad-CAM on spatial features
        gradients = spatial_feats.grad  # [1, C, H, W]
        if gradients is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        weights = gradients.mean(dim=[2, 3])  # [1, C]
        cam = (weights[:, :, None, None] * spatial_feats.detach()).sum(dim=1)  # [1, H, W]
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Public API ──────────────────────────────────────

_gradcam_runners = {
    "teacher": _SwinGradCAM(),
    "dkd": _ResNetGradCAM("dkd"),
    "takd": _ResNetGradCAM("takd"),
    "baseline": _ResNetGradCAM("baseline"),
}


def generate_gradcam(
    input_tensor: torch.Tensor,
    class_idx: int,
    model_role: Optional[str] = None,
) -> dict:
    """Generate GradCAM heatmap.

    Args:
        input_tensor: Preprocessed image tensor [1, 3, 224, 224]
        class_idx: Target class ID for GradCAM
        model_role: Specific model or None for all 4

    Returns:
        dict with model_role → heatmap as numpy array, or "all" → dict
    """
    if model_role:
        runner = _gradcam_runners.get(model_role)
        if runner is None:
            return {"error": f"Unknown role: {model_role}"}
        cam = runner(input_tensor, class_idx)
        return {model_role: cam}

    results = {}
    for role, runner in _gradcam_runners.items():
        try:
            results[role] = runner(input_tensor, class_idx)
        except Exception as e:
            print(f"  [gradcam:{role}] Error: {e}")
            results[role] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return results


def heatmap_to_base64(cam: np.ndarray, original_image_bytes: bytes) -> str:
    """Overlay heatmap on original image and return base64 PNG."""
    import io, base64
    original = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    original = original.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Create jet colormap heatmap
    h = np.clip(cam * 255, 0, 255).astype(np.uint8)
    colored = _apply_jet_colormap(h)
    heatmap_pil = Image.fromarray(colored)

    blended = Image.blend(original, heatmap_pil, alpha=0.5)

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _apply_jet_colormap(gray: np.ndarray):
    """Simple jet colormap: 0=blue, 128=green, 255=red."""
    h, w = gray.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[:, :, 2] = np.clip(255 * (1 - 2 * np.abs(gray / 255.0 - 0.25)), 0, 255)
    colored[:, :, 1] = np.clip(255 * (1 - 2 * np.abs(gray / 255.0 - 0.5)), 0, 255)
    colored[:, :, 0] = np.clip(255 * (1 - 2 * np.abs(gray / 255.0 - 0.75)), 0, 255)
    return colored
