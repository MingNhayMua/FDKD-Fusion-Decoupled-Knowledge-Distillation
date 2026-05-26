"""
Decoupled Knowledge Distillation decomposition.

Reference: Zhao et al., "Decoupled Knowledge Distillation" (CVPR 2022)
"""
import numpy as np

from utils.config import NUM_CLASSES, MODEL_ROLES
from utils.distributions import logits_to_probs
from utils.math_utils import kl_divergence, rank_correlation
from utils.labels import get_label


def _make_ranking(dist, indices, top_idx):
    return [
        {"class": get_label(int(indices[i])),
         "class_id": int(indices[i]),
         "prob": float(dist[i])}
        for i in top_idx
    ]


def compute_dkd(logits_dict: dict, temperature: float = 1.0) -> dict:
    """Compute full DKD decomposition across all models."""
    if not all(k in logits_dict for k in MODEL_ROLES):
        return {"error": "Not all models available"}

    probs = logits_to_probs(logits_dict, temperature)
    target = int(np.argmax(probs["teacher"]))

    # TCKD: target class confidences
    tckd = {
        "target_class": get_label(target),
        "target_class_id": target,
        "confidences": {k: float(probs[k][target]) for k in MODEL_ROLES},
    }
    # Pairwise alignments
    for i, k1 in enumerate(MODEL_ROLES):
        for k2 in MODEL_ROLES[i + 1:]:
            tckd[f"{k1}_{k2}_alignment"] = float(
                1 - abs(probs[k1][target] - probs[k2][target]))

    # NCKD: non-target class distributions
    mask = np.ones(NUM_CLASSES, dtype=bool)
    mask[target] = False
    nt_indices = np.where(mask)[0]

    nt = {}
    for key in MODEL_ROLES:
        raw = probs[key][mask]
        nt[key] = raw / (raw.sum() + 1e-10)

    nt_top = np.argsort(nt["teacher"])[::-1][:10]

    nckd = {}
    for key in MODEL_ROLES:
        nckd[f"{key}_ranking"] = _make_ranking(nt[key], nt_indices, nt_top)

    # Pairwise KL & rank correlation
    for i, k1 in enumerate(MODEL_ROLES):
        for k2 in MODEL_ROLES[i + 1:]:
            nckd[f"kl_{k1}_{k2}"] = kl_divergence(nt[k1], nt[k2])
            nckd[f"rank_correlation_{k1}_{k2}"] = rank_correlation(nt[k1], nt[k2])

    # Dark knowledge: teacher's non-target structure
    dark_knowledge = {}
    for key in MODEL_ROLES:
        dark_knowledge[f"{key}_top_non_target"] = _make_ranking(
            nt[key], nt_indices, nt_top)
    dark_knowledge["explanation"] = (
        "Dark knowledge represents the semantic relationships encoded in "
        "each model's non-target class probabilities. Higher probabilities "
        "for visually similar classes indicate learned inter-class structure."
    )

    return {"tckd": tckd, "nckd": nckd, "dark_knowledge": dark_knowledge}
