"""
Research metrics: KL divergence, cosine similarity, entropy.
"""
from utils.config import MODEL_ROLES
from utils.distributions import logits_to_probs
from utils.math_utils import kl_divergence, cosine_similarity, entropy


def compute_metrics(logits_dict: dict, temperature: float = 1.0) -> dict:
    """Compute distribution-level metrics across all model pairs."""
    if not all(k in logits_dict for k in MODEL_ROLES):
        return {"error": "Not all models available"}

    probs = logits_to_probs(logits_dict, temperature)
    t, a, s = probs["teacher"], probs["assistant"], probs["student"]

    return {
        "kl_teacher_student": kl_divergence(t, s),
        "kl_teacher_assistant": kl_divergence(t, a),
        "kl_assistant_student": kl_divergence(a, s),
        "cosine_teacher_student": cosine_similarity(t, s),
        "cosine_teacher_assistant": cosine_similarity(t, a),
        "cosine_assistant_student": cosine_similarity(a, s),
        "entropy_teacher": entropy(t),
        "entropy_assistant": entropy(a),
        "entropy_student": entropy(s),
    }
