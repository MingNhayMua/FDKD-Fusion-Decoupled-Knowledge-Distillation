"""
Full Teacher → Assistant → Student inference pipeline.

MMPretrain's ImageClassifier.forward() in 'tensor' mode returns the
logits tensor directly — we use this mode for clean integration.
"""
import torch

from utils.config import MODEL_INFO, MODEL_ROLES
from backend.models.loader import MODELS
from utils.distributions import extract_predictions


def _get_logits(model, input_tensor: torch.Tensor) -> torch.Tensor:
    """Extract logits from an MMPretrain ImageClassifier.

    MMPretrain models support multiple forward modes:
    - model(inputs, mode='tensor')  → raw backbone+neck+head tensor output
    - model(inputs, mode='predict') → list of DataSample with predictions

    We use a direct path: backbone → neck → head.fc to get logits.
    """
    try:
        # Direct path through MMPretrain architecture components
        # This avoids any DataSample/DataPreprocessor issues
        x = model.backbone(input_tensor)
        # backbone returns tuple of feature maps; take the last one
        if isinstance(x, (tuple, list)):
            x = x[-1]
        x = model.neck(x)
        # neck output might be tuple
        if isinstance(x, (tuple, list)):
            x = x[-1]
        # Get logits from head's FC layer directly
        logits = model.head.fc(x)
        return logits
    except Exception as e:
        # Fallback: try calling model directly
        print(f"  ⚠️ Direct path failed ({e}), trying model(tensor)")
        output = model(input_tensor)
        if isinstance(output, torch.Tensor):
            return output
        # If it returns DataSample list, extract logits
        if isinstance(output, (list, tuple)) and hasattr(output[0], 'pred_score'):
            return output[0].pred_score.unsqueeze(0)
        raise RuntimeError(f"Cannot extract logits from model output: {type(output)}")


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

            logits = _get_logits(model, input_tensor)
            logits_dict[key] = logits

            preds = extract_predictions(logits, temperature)
            preds.update(MODEL_INFO[key])
            predictions[key] = preds

    return predictions, logits_dict
