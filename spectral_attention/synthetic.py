"""Canonical causal row-stochastic attention maps, used in the examples and tests."""

from typing import Optional

import numpy as np


def copy_map(n: int) -> np.ndarray:
    """Every token attends to itself: A = I (doubly stochastic, full routing)."""
    return np.eye(n)


def sink_map(n: int) -> np.ndarray:
    """Every token attends to the first one: A = 1 e_1^T (full collapse)."""
    A = np.zeros((n, n))
    A[:, 0] = 1.0
    return A


def shift_map(n: int) -> np.ndarray:
    """Previous-token head: A[i, i-1] = 1, the first token attends to itself."""
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    A[np.arange(1, n), np.arange(n - 1)] = 1.0
    return A


def local_map(n: int, tau: float = 2.5) -> np.ndarray:
    """Causal exponential-decay attention A[i, j] ∝ exp(-(i - j) / tau)."""
    i, j = np.indices((n, n))
    A = np.where(j <= i, np.exp(-(i - j) / tau), 0.0)
    return A / A.sum(axis=1, keepdims=True)


def interpolate(A_from: np.ndarray, A_to: np.ndarray, alpha: float) -> np.ndarray:
    """(1 - alpha) A_from + alpha A_to; stays causal and row-stochastic."""
    return (1.0 - alpha) * A_from + alpha * A_to


def random_causal(n: int, temperature: float = 1.0, sink_logit: float = 0.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Softmax of Gaussian logits under a causal mask, with an optional bonus on token 0."""
    rng = np.random.default_rng() if rng is None else rng
    logits = rng.standard_normal((n, n)) / temperature
    logits[:, 0] += sink_logit
    logits[np.triu_indices(n, k=1)] = -np.inf
    logits -= logits.max(axis=1, keepdims=True)
    A = np.exp(logits)
    return A / A.sum(axis=1, keepdims=True)
