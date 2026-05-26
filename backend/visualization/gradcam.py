"""
Grad-CAM visualization for Teacher / Assistant / Student.

Teacher: pure-PyTorch Swin (backbone.stages.X.blocks.Y)
Assistant/Student: torchvision ResNet (layer4)
"""
import io
import base64
import numpy as np
import torch
from PIL import Image

from backend.models.loader import MODELS


def generate_gradcam_all(input_tensor: torch.Tensor, img_np: np.ndarray) -> dict:
    """Generate Grad-CAM heatmaps for all loaded models."""
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
                # Pure-PyTorch Swin: use last block's norm2
                target_layers = [model.backbone.stages[-1].blocks[-1].norm2]

                def reshape_transform(tensor, height=7, width=7):
                    # (B, num_tokens, C) → (B, C, H, W)
                    return tensor.permute(0, 2, 1).reshape(
                        tensor.size(0), tensor.size(2), height, width
                    )

                cam = GradCAM(
                    model=model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform,
                )
            else:
                # torchvision ResNet
                target_layers = [model.layer4[-1]]
                cam = GradCAM(model=model, target_layers=target_layers)

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            visualization = show_cam_on_image(
                img_np, grayscale_cam[0, :], use_rgb=True
            )

            img_pil = Image.fromarray(visualization)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            results[key] = base64.b64encode(buffer.getvalue()).decode("utf-8")

        except Exception as e:
            results[key] = None
            print(f"Grad-CAM failed for {key}: {e}")

    return results
