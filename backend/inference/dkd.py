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
    """Compute full DKD decomposition across Teacher/Assistant/Student."""
    if not all(k in logits_dict for k in MODEL_ROLES):
        return {"error": "Not all models available"}

    probs = logits_to_probs(logits_dict, temperature)
    t, a, s = probs["teacher"], probs["assistant"], probs["student"]
    target = int(np.argmax(t))

    tckd = {
        "target_class": get_label(target),
        "target_class_id": target,
        "teacher_confidence": float(t[target]),
        "assistant_confidence": float(a[target]),
        "student_confidence": float(s[target]),
        "ta_alignment": float(1 - abs(t[target] - a[target])),
        "as_alignment": float(1 - abs(a[target] - s[target])),
        "ts_alignment": float(1 - abs(t[target] - s[target])),
    }

    mask = np.ones(NUM_CLASSES, dtype=bool)
    mask[target] = False
    nt_indices = np.where(mask)[0]

    nt = {}
    for key in MODEL_ROLES:
        raw = probs[key][mask]
        nt[key] = raw / (raw.sum() + 1e-10)

    nt_top = np.argsort(nt["teacher"])[::-1][:10]

    nckd = {
        "teacher_ranking": _make_ranking(nt["teacher"], nt_indices, nt_top),
        "assistant_ranking": _make_ranking(nt["assistant"], nt_indices, nt_top),
        "student_ranking": _make_ranking(nt["student"], nt_indices, nt_top),
        "kl_ta": kl_divergence(nt["teacher"], nt["assistant"]),
        "kl_as": kl_divergence(nt["assistant"], nt["student"]),
        "kl_ts": kl_divergence(nt["teacher"], nt["student"]),
        "rank_correlation_ta": rank_correlation(nt["teacher"], nt["assistant"]),
        "rank_correlation_as": rank_correlation(nt["assistant"], nt["student"]),
        "rank_correlation_ts": rank_correlation(nt["teacher"], nt["student"]),
    }

    dark_knowledge = {
        "teacher_top_non_target": _make_ranking(nt["teacher"], nt_indices, nt_top),
        "student_top_non_target": _make_ranking(nt["student"], nt_indices, nt_top),
        "explanation": (
            "Dark knowledge represents the semantic relationships encoded in "
            "the teacher's non-target class probabilities. Higher probabilities "
            "for visually similar classes indicate the teacher has learned "
            "meaningful inter-class structure that transfers to the student."
        ),
    }

    return {"tckd": tckd, "nckd": nckd, "dark_knowledge": dark_knowledge}
