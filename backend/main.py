"""
FDKD Demo — FastAPI Application

Routes:
  GET  /api/health         Server status + model info
  POST /api/inference       Full T→A→S pipeline
  POST /api/distribution    Recompute at new temperature
  POST /api/gradcam         Generate Grad-CAM heatmaps
"""
import uuid
import base64

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.image import preprocess_image
from backend.models.loader import MODELS
from backend.inference.pipeline import run_inference
from backend.inference.dkd import compute_dkd
from backend.inference.metrics import compute_metrics
from backend.visualization.gradcam import generate_gradcam_all


app = FastAPI(
    title="FDKD Interactive Demo API",
    description="Fusion Decoupled Knowledge Distillation visualization backend",
    version="1.0.0",
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
    from utils.config import DEVICE, NUM_CLASSES
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": {k: v is not None for k, v in MODELS.items()},
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
        "teacher": preds.get("teacher", {}),
        "assistant": preds.get("assistant", {}),
        "student": preds.get("student", {}),
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
        "teacher": preds.get("teacher", {}),
        "assistant": preds.get("assistant", {}),
        "student": preds.get("student", {}),
        "dkd": dkd,
        "metrics": metrics,
    })


@app.post("/api/gradcam")
async def gradcam(file: UploadFile = File(...)):
    """Generate Grad-CAM heatmaps for all three models."""
    image_bytes = await file.read()
    input_tensor, img_np = preprocess_image(image_bytes)
    results = generate_gradcam_all(input_tensor, img_np)
    return JSONResponse({"gradcam": results})
