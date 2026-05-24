"""
Full Teacher → Assistant → Student inference pipeline.
"""
import torch

from utils.config import MODEL_INFO, MODEL_ROLES
from backend.models.loader import MODELS
from utils.distributions import extract_predictions


def run_inference(input_tensor: torch.Tensor, temperature: float = 1.0) -> tuple[dict, dict]:
    """Run inference through all three models."""
    predictions = {}
    logits_dict = {}

    with torch.no_grad():
        for key in MODEL_ROLES:
            model = MODELS.get(key)
            if model is None:
                predictions[key] = {"error": f"{key} model not loaded"}
                continue

            logits = model(input_tensor)
            logits_dict[key] = logits

            preds = extract_predictions(logits, temperature)
            preds.update(MODEL_INFO[key])
            predictions[key] = preds

    return predictions, logits_dict
