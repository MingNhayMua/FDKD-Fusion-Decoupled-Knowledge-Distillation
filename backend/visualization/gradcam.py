"""
Grad-CAM visualization for Teacher / Assistant / Student.

Uses torchvision (ResNet) and timm (Swin-B) models directly.
"""
import io
import base64
import numpy as np
import torch
from PIL import Image

from backend.models.loader import MODELS


def generate_gradcam_all(input_tensor: torch.Tensor, img_np: np.ndarray) -> dict:
    """Generate Grad-CAM heatmaps for all loaded models.

    Returns dict of base64-encoded PNG overlay images per model.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        return {"error": "pytorch-grad-cam not installed"}

    results = {}

    for key, model in MODELS.items():
        if model is None:
            continue

        try:
            # Target layer differs per architecture
            if key == "teacher":
                # timm Swin-B: last layer's last block
                target_layers = [model.layers[-1].blocks[-1].norm1]
            else:
                # torchvision ResNet: last conv block
                target_layers = [model.layer4[-1]]

            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            visualization = show_cam_on_image(
                img_np, grayscale_cam[0, :], use_rgb=True
            )

            # Encode to base64 PNG
            img_pil = Image.fromarray(visualization)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            results[key] = base64.b64encode(buffer.getvalue()).decode("utf-8")

        except Exception as e:
            results[key] = None
            print(f"Grad-CAM failed for {key}: {e}")

    return results
