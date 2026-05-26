"""
GradCAM-style saliency maps for all 4 FDKD models using traced models.

Uses gradient-based approach with fallback to occlusion for robust operation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from backend.models.loader import MODELS
from utils.config import DEVICE, IMAGE_SIZE, MODEL_ROLES


def generate_gradcam(input_tensor: torch.Tensor, class_idx: int) -> dict:
    """Generate heatmap for all 4 models.

    Tries gradient-based first, falls back to occlusion if backward fails.
    """
    results = {}
    for role in MODEL_ROLES:
        model = MODELS.get(role)
        if model is None:
            results[role] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            continue
        try:
            results[role] = _compute_heatmap(model, input_tensor, class_idx)
        except Exception as e:
            print(f"  [gradcam:{role}] {e}, using fallback")
            try:
                results[role] = _occlusion_map(model, input_tensor, class_idx)
            except Exception as e2:
                print(f"  [gradcam:{role}] Fallback also failed: {e2}")
                results[role] = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5
    return results


def _compute_heatmap(model, x_input, class_idx):
    """Try gradient saliency first."""
    x = x_input.clone().to(DEVICE).requires_grad_(True)
    output = model(x)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    score = output[0, class_idx]
    score.backward()

    if x.grad is None:
        raise RuntimeError("Gradient is None")

    grad = x.grad.abs().max(dim=1)[0].squeeze(0).detach().cpu().numpy()
    gm = grad.max()
    if gm < 1e-10:
        raise RuntimeError("All-zero gradient")

    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return _smooth(grad)


def _occlusion_map(model, x_input, class_idx, window=16, stride=8):
    """Occlusion sensitivity: measure score drop when masking patches (no gradients needed)."""
    H, W = IMAGE_SIZE, IMAGE_SIZE
    x = x_input.clone().to(DEVICE)
    output = model(x)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    base_score = output[0, class_idx].item()

    heatmap = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    # Use gray occlusion
    occluder = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)

    for y in range(0, H - window + 1, stride):
        for x0 in range(0, W - window + 1, stride):
            x_occ = x.clone()
            x_occ[:, :, y:y + window, x0:x0 + window] = occluder
            with torch.no_grad():
                out = model(x_occ)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                drop = base_score - out[0, class_idx].item()
            heatmap[y:y + window, x0:x0 + window] += max(0, drop)
            count[y:y + window, x0:x0 + window] += 1

    count[count < 1] = 1
    heatmap /= count

    hm = heatmap.max()
    if hm > 1e-8:
        heatmap = (heatmap - heatmap.min()) / (hm - heatmap.min() + 1e-8)
    return _smooth(heatmap)


def _smooth(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur for smoother visualization."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)
