"""
Global configuration for the FDKD Demo Backend.
"""
import os
import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
NUM_CLASSES = 200

# Paths — update these for your environment
CHECKPOINT_DIR = os.environ.get(
    "FDKD_CHECKPOINT_DIR",
    "/content/drive/MyDrive/checkpoints"
)

# FDKD pipeline checkpoint mapping (directory name → model role)
# Priority order: first match wins
TEACHER_DIRS = ["swinb_fully"]
ASSISTANT_DIRS = ["dkd_swinb_r152", "r152_fully"]
STUDENT_DIRS = ["dkd_r152_r18", "dkd_swinb_r18", "r18_fully"]

# Checkpoint file extensions to search for
CHECKPOINT_EXTENSIONS = ('.pth', '.pt', '.pkl', '.bin', '.ckpt', '.pth.tar')

# Image preprocessing
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model metadata (shared by pipeline, metrics display, etc.)
MODEL_INFO = {
    "teacher":   {"name": "Swin-B",    "params": "86.95M", "gflops": "15,467"},
    "assistant": {"name": "ResNet-152", "params": "58.55M", "gflops": "11,557"},
    "student":   {"name": "ResNet-18",  "params": "11.28M", "gflops": "1,819"},
}

MODEL_ROLES = ["teacher", "assistant", "student"]

# ngrok
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
