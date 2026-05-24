"""
Softmax, temperature scaling, and top-k extraction.
"""
import numpy as np
import torch
import torch.nn.functional as F

from utils.config import MODEL_ROLES
from utils.labels import get_label


def temperature_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute softmax with temperature: p = softmax(z / T)."""
    return F.softmax(logits / temperature, dim=-1)


def logits_to_probs(logits_dict: dict, temperature: float = 1.0) -> dict:
    """Convert logits dict → numpy probability arrays for all models."""
    probs = {}
    for key in MODEL_ROLES:
        if key in logits_dict:
            p = temperature_softmax(logits_dict[key], temperature)
            probs[key] = p.cpu().numpy().flatten()
    return probs


def extract_predictions(logits: torch.Tensor, temperature: float = 1.0, topk: int = 10) -> dict:
    """Extract top-k predictions, confidence, and entropy from logits."""
    probs = temperature_softmax(logits, temperature)
    probs_np = probs.cpu().numpy().flatten()

    top_indices = np.argsort(probs_np)[::-1][:topk]
    topk_list = [
        {
            "class_id": int(idx),
            "class": get_label(int(idx)),
            "prob": float(probs_np[idx]),
        }
        for idx in top_indices
    ]

    entropy = float(-np.sum(probs_np * np.log(probs_np + 1e-10)))

    return {
        "topk": topk_list,
        "confidence": float(probs_np[top_indices[0]]),
        "predicted_class": get_label(int(top_indices[0])),
        "predicted_class_id": int(top_indices[0]),
        "entropy": entropy,
        "full_probs": probs_np.tolist(),
    }
