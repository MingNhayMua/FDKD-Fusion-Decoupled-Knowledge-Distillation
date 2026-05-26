"""
Research metrics: KL divergence, cosine similarity, entropy.
"""
from utils.config import MODEL_ROLES
from utils.distributions import logits_to_probs
from utils.math_utils import kl_divergence, cosine_similarity, entropy


def compute_metrics(logits_dict: dict, temperature: float = 1.0) -> dict:
    """Compute distribution-level metrics across all models."""
    if not all(k in logits_dict for k in MODEL_ROLES):
        return {"error": "Not all models available"}

    probs = logits_to_probs(logits_dict, temperature)

    result = {}
    # Entropy per model
    for key in MODEL_ROLES:
        result[f"entropy_{key}"] = entropy(probs[key])

    # Pairwise KL & cosine
    for i, k1 in enumerate(MODEL_ROLES):
        for k2 in MODEL_ROLES[i + 1:]:
            result[f"kl_{k1}_{k2}"] = kl_divergence(probs[k1], probs[k2])
            result[f"cosine_{k1}_{k2}"] = cosine_similarity(probs[k1], probs[k2])

    return result
