"""
Grad-CAM visualization for Teacher / Assistant / Student.

Uses torchvision (ResNet) and timm (Swin-B) models directly.
For Swin Transformers, we use a reshape wrapper to convert
token-based (B, N, C) output to spatial (B, C, H, W) for Grad-CAM.
"""
import io
import base64
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from backend.models.loader import MODELS


class _SwinNormReshape(nn.Module):
    """Wrapper that reshapes Swin's (B, N, C) norm output to (B, C, H, W).

    Grad-CAM expects 2D spatial feature maps (B, C, H, W), but Swin's
    LayerNorm outputs (B, num_tokens, channels). This wrapper converts
    between the two formats.
    """
    def __init__(self, norm_layer):
        super().__init__()
        self.norm = norm_layer

    def forward(self, x):
        # x is (B, N, C) from Swin
        out = self.norm(x)
        B, N, C = out.shape
        H = W = int(math.sqrt(N))
        # Reshape to (B, C, H, W) for Grad-CAM
        return out.permute(0, 2, 1).reshape(B, C, H, W)


def _get_swin_target_layer(model):
    """Get a Grad-CAM compatible target layer for timm Swin.

    Returns the final norm layer wrapped with spatial reshape.
    """
    try:
        # timm Swin: model.norm is the final LayerNorm
        return [model.norm]
    except AttributeError:
        try:
            return [model.layers[-1].blocks[-1].norm2]
        except (AttributeError, IndexError):
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
            if key == "teacher":
                target_layers = _get_swin_target_layer(model)
                if target_layers is None:
                    results[key] = None
                    continue

                # Swin needs reshape_transform for Grad-CAM
                def reshape_transform(tensor, height=7, width=7):
                    # tensor shape: (B, num_tokens, C)
                    result = tensor.permute(0, 2, 1)  # (B, C, N)
                    result = result.reshape(
                        result.size(0), result.size(1), height, width
                    )
                    return result

                cam = GradCAM(
                    model=model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform,
                )
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
