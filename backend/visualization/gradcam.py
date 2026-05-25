"""
Grad-CAM visualization for Teacher / Assistant / Student.

Updated for MMPretrain model architecture where backbone is accessed
via model.backbone.* instead of model.layer4.* (torchvision).
"""
import io
import base64
import numpy as np
import torch
from PIL import Image

from backend.models.loader import MODELS
from backend.inference.pipeline import _get_logits


class _MMPretrainWrapper(torch.nn.Module):
    """Wrapper to make MMPretrain model compatible with pytorch-grad-cam.

    pytorch-grad-cam expects model(input) → logits tensor,
    but MMPretrain's forward() returns DataSample objects.
    This wrapper uses the direct backbone→neck→head.fc path.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return _get_logits(self.model, x)


def _get_target_layers(model, key: str):
    """Get the appropriate target layer for Grad-CAM.

    For MMPretrain models, the backbone is accessed via model.backbone.*
    """
    backbone = model.backbone

    if key == "teacher":
        # Swin Transformer: last stage, last block
        try:
            return [backbone.stages[-1].blocks[-1].norm1]
        except (AttributeError, IndexError):
            try:
                return [backbone.layers[-1].blocks[-1].norm1]
            except (AttributeError, IndexError):
                print(f"  ⚠️ Cannot find Swin target layer")
                return None
    else:
        # ResNet: last layer4 block
        try:
            return [backbone.layer4[-1]]
        except (AttributeError, IndexError):
            print(f"  ⚠️ Cannot find ResNet target layer")
            return None


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
            target_layers = _get_target_layers(model, key)
            if target_layers is None:
                results[key] = None
                continue

            # Wrap model so GradCAM gets logits from forward()
            wrapper = _MMPretrainWrapper(model)
            cam = GradCAM(model=wrapper, target_layers=target_layers)
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
