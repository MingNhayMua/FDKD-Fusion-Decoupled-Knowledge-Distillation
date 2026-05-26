"""
FDKD Demo — FastAPI Application

Routes:
  GET  /api/health         Server status + model info
  POST /api/inference       Full T→A→S pipeline
  POST /api/distribution    Recompute at new temperature
  POST /api/gradcam         GradCAM heatmaps for all models
"""
import uuid
import base64
import io

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from utils.image import preprocess_image
from backend.models.loader import MODELS
from backend.inference.pipeline import run_inference
from backend.inference.dkd import compute_dkd
from backend.inference.metrics import compute_metrics
from backend.visualization.gradcam import generate_gradcam


app = FastAPI(
    title="FDKD Interactive Demo API",
    description="Fusion Decoupled Knowledge Distillation visualization backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Server health check and model status."""
    from utils.config import DEVICE, NUM_CLASSES, MODEL_ROLES, MODEL_INFO
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": {k: v is not None for k, v in MODELS.items()},
        "model_roles": MODEL_ROLES,
        "model_info": MODEL_INFO,
        "num_classes": NUM_CLASSES,
    }


@app.post("/api/inference")
async def inference(
    file: UploadFile = File(...),
    temperature: float = Form(default=1.0),
):
    """Full pipeline: upload image → T/A/S predictions + DKD + metrics."""
    image_bytes = await file.read()
    input_tensor, _ = preprocess_image(image_bytes)

    preds, logits_dict = run_inference(input_tensor, temperature)
    dkd = compute_dkd(logits_dict, temperature)
    metrics = compute_metrics(logits_dict, temperature)
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    return JSONResponse({
        "image_id": str(uuid.uuid4()),
        "image_base64": img_b64,
        "temperature": temperature,
        "models": preds,
        "dkd": dkd,
        "metrics": metrics,
    })


@app.post("/api/distribution")
async def recompute_distribution(
    file: UploadFile = File(...),
    temperature: float = Form(default=1.0),
):
    """Recompute distributions at a new temperature."""
    image_bytes = await file.read()
    input_tensor, _ = preprocess_image(image_bytes)

    preds, logits_dict = run_inference(input_tensor, temperature)
    dkd = compute_dkd(logits_dict, temperature)
    metrics = compute_metrics(logits_dict, temperature)

    return JSONResponse({
        "temperature": temperature,
        "models": preds,
        "dkd": dkd,
        "metrics": metrics,
    })


@app.post("/api/gradcam")
async def gradcam(file: UploadFile = File(...)):
    """Generate GradCAM heatmaps for all 4 models."""
    image_bytes = await file.read()
    input_tensor, _ = preprocess_image(image_bytes)

    preds, logits_dict = run_inference(input_tensor, 1.0)
    class_idx = preds.get("teacher", {}).get("predicted_class_id", 0)

    heatmaps = generate_gradcam(input_tensor, class_idx)

    heatmap_images = {}
    for role, cam in heatmaps.items():
        heatmap_images[role] = _overlay_heatmap(image_bytes, cam)

    return JSONResponse({
        "target_class": class_idx,
        "heatmaps": heatmap_images,
    })


def _overlay_heatmap(image_bytes: bytes, cam) -> str:
    """Overlay heatmap on original image using PIL, return base64 PNG."""
    import numpy as np

    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original = original.resize((224, 224))

    # Create jet colormap heatmap
    h = cam * 255
    h = np.clip(h, 0, 255).astype(np.uint8)
    heatmap = _apply_jet_colormap(h)

    # Blend
    heatmap_pil = Image.fromarray(heatmap).resize((224, 224))
    blended = Image.blend(original, heatmap_pil, alpha=0.5)

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return base64.b64encode(buf.read()).decode("utf-8")


def _apply_jet_colormap(gray: np.ndarray):
    """Simple jet colormap: 0=blue, 128=green, 255=red."""
    import numpy as np
    h, w = gray.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    # Blue channel: 1 - 2*|x - 0.25|
    colored[:, :, 2] = np.clip(255 * (1 - 2 * np.abs(gray / 255.0 - 0.25)), 0, 255)
    # Green channel: 1 - 2*|x - 0.5|  
    colored[:, :, 1] = np.clip(255 * (1 - 2 * np.abs(gray / 255.0 - 0.5)), 0, 255)
    # Red channel: 1 - 2*|x - 0.75|
    colored[:, :, 0] = np.clip(255 * (1 - 2 * np.abs(gray / 255.0 - 0.75)), 0, 255)
    return colored
