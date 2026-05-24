"""
Shared mathematical utilities for distribution analysis.
"""
import numpy as np
from scipy.stats import kendalltau
from scipy.spatial.distance import cosine as cosine_dist


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) — measures how Q diverges from P."""
    return float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Cosine similarity between two distributions."""
    return float(1 - cosine_dist(p, q))


def entropy(p: np.ndarray) -> float:
    """Shannon entropy H(P) = -Σ p·log(p)."""
    return float(-np.sum(p * np.log(p + 1e-10)))


def rank_correlation(p: np.ndarray, q: np.ndarray) -> float:
    """Kendall τ rank correlation between two distributions."""
    tau, _ = kendalltau(np.argsort(p)[::-1], np.argsort(q)[::-1])
    return float(tau) if not np.isnan(tau) else 0.0
