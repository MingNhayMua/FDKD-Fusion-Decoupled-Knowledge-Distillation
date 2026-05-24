"""
Image preprocessing utilities.
"""
import io
import numpy as np
from PIL import Image
from torchvision import transforms

from utils.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEVICE


TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image_bytes: bytes) -> tuple:
    """Preprocess uploaded image.

    Returns:
        (tensor, numpy_rgb): input tensor on DEVICE, and float32 RGB numpy
                             array normalized to [0,1] for Grad-CAM overlay.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.float32(img_resized) / 255.0
    return tensor, img_np
