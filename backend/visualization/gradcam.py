"""
GradCAM-style visualization for all 4 FDKD models using traced models.

Uses input gradient saliency (works on any model including torch.jit traced).
Does NOT require raw checkpoint loading — uses the same traced .pt files.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from backend.models.loader import MODELS
from utils.config import DEVICE, IMAGE_SIZE, MODEL_ROLES


def generate_gradcam(
    input_tensor: torch.Tensor,
    class_idx: int,
) -> dict:
    """Generate input-gradient heatmap for all 4 models.

    Computes gradient of target class score w.r.t. input pixels.
    Works on traced (torch.jit) models — no layer hooking needed.

    Args:
        input_tensor: Preprocessed image [1, 3, 224, 224]
        class_idx: Target class ID

    Returns:
        dict: role → numpy heatmap array (H, W)
    """
    results = {}
    for role in MODEL_ROLES:
        model = MODELS.get(role)
        if model is None:
            results[role] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            continue

        try:
            results[role] = _saliency_map(model, input_tensor, class_idx)
        except Exception as e:
            print(f"  [gradcam:{role}] Error: {e}")
            results[role] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    return results


def _saliency_map(model, input_tensor, class_idx):
    """Compute input saliency map: |∂score/∂input|, max over channels."""
    x = input_tensor.clone().to(DEVICE)
    x.requires_grad = True

    output = model(x)
    score = output[0, class_idx]
    score.backward()

    saliency = x.grad.abs().max(dim=1)[0]  # [1, H, W]
    saliency = saliency.squeeze(0).detach().cpu().numpy()

    # Normalize
    smin, smax = saliency.min(), saliency.max()
    if smax - smin > 1e-8:
        saliency = (saliency - smin) / (smax - smin)

    # Smooth with small Gaussian blur for better visualization
    saliency = _gaussian_blur(saliency, sigma=2.0)

    return saliency


def _gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian blur to smooth the heatmap."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)


def heatmap_to_base64(cam: np.ndarray, original_image_bytes: bytes) -> str:
    """Overlay heatmap on original image and return base64 PNG."""
    import io, base64

    original = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    original = original.resize((IMAGE_SIZE, IMAGE_SIZE))

    h = np.clip(cam * 255, 0, 255).astype(np.uint8)
    colored = _apply_jet_colormap(h)
    heatmap_pil = Image.fromarray(colored)

    blended = Image.blend(original, heatmap_pil, alpha=0.5)

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _apply_jet_colormap(gray: np.ndarray):
    """Simple jet colormap: 0=blue, 1=red."""
    h, w = gray.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    x = gray / 255.0
    colored[:, :, 2] = np.clip(255 * (1 - 2 * np.abs(x - 0.25)), 0, 255)
    colored[:, :, 1] = np.clip(255 * (1 - 2 * np.abs(x - 0.5)), 0, 255)
    colored[:, :, 0] = np.clip(255 * (1 - 2 * np.abs(x - 0.75)), 0, 255)
    return colored
