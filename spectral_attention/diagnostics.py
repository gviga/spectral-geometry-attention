"""
Per-head attention diagnostics compared in Sec. 4 of the paper.

Every function takes one causal row-stochastic attention map A (n x n) and
returns a scalar. All scores except `log_det_score` lie in [0, 1], so they can
be plotted on a common axis.

Raw attention statistics
    sink_score            mean attention mass on the first token
    prev_token_score      mean A[i, i-1] (RAUQ's head-selection signal)
    attention_entropy     mean row entropy, normalised by log n
    column_sum_entropy    exp H(mu) / n, effective support of the column mass

Counting-geometry spectral diagnostics
    log_det_score         (1/n) log det A = mean log A_ii (AttentionScore)
    laplacian_diag_entropy exp H(d) / n of the causal Laplacian diagonal (LapEigvals)
    euclidean_pr          PR(A^T A)

Intrinsic TDO diagnostics (ours)
    intrinsic_pr          PR(D_{mu,pi})
    intrinsic_pr_centered centered PR(D_{mu,pi})
"""

from typing import Dict, Optional

import numpy as np

from .tdo import spectral_entropy, tdo_pr, tdo_pr_centered

_EPS = 1e-12


def _entropy(p: np.ndarray) -> float:
    p = p[p > _EPS]
    return float(-np.sum(p * np.log(p)))


def sink_score(A: np.ndarray) -> float:
    return float(A[:, 0].mean())


def prev_token_score(A: np.ndarray) -> float:
    return float(np.diag(A, -1).mean()) if A.shape[0] > 1 else 0.0


def attention_entropy(A: np.ndarray) -> float:
    n = A.shape[0]
    if n < 2:
        return 0.0
    return float(np.mean([_entropy(row) for row in A]) / np.log(n))


def column_sum_entropy(A: np.ndarray) -> float:
    mu = A.mean(axis=0)
    return float(np.exp(_entropy(mu / mu.sum())) / A.shape[0])


def log_det_score(A: np.ndarray) -> float:
    return float(np.mean(np.log(np.maximum(np.diag(A), _EPS))))


def laplacian_diag_entropy(A: np.ndarray) -> float:
    n = A.shape[0]
    n_queries = np.arange(n, 0, -1, dtype=float)       # queries that can see key j
    d = np.maximum(A.sum(axis=0) / n_queries - np.diag(A), 0.0)
    if d.sum() <= _EPS:
        return 0.0
    return float(np.exp(_entropy(d / d.sum())) / n)


def euclidean_pr(A: np.ndarray) -> float:
    return tdo_pr(A, "euclidean")


def intrinsic_pr(A: np.ndarray) -> float:
    return tdo_pr(A, "intrinsic")


def intrinsic_pr_centered(A: np.ndarray) -> float:
    return tdo_pr_centered(A)


DIAGNOSTICS = {
    "sink_score": sink_score,
    "column_sum_entropy": column_sum_entropy,
    "attention_entropy": attention_entropy,
    "prev_token_score": prev_token_score,
    "log_det_score": log_det_score,
    "laplacian_diag_entropy": laplacian_diag_entropy,
    "euclidean_pr": euclidean_pr,
    "intrinsic_pr": intrinsic_pr,
    "intrinsic_pr_centered": intrinsic_pr_centered,
}

LABELS = {
    "sink_score": "Sink score",
    "column_sum_entropy": "Column-sum entropy",
    "attention_entropy": "Attention entropy",
    "prev_token_score": "Prev. token",
    "log_det_score": "Log-det",
    "laplacian_diag_entropy": "LapEigval entropy",
    "euclidean_pr": r"PR($A^\top A$)",
    "intrinsic_pr": r"PR($D_{\mu,\pi}$) (ours)",
    "intrinsic_pr_centered": r"PR$_\mathrm{cent}$($D_{\mu,\pi}$) (ours)",
}


def head_diagnostics(A: np.ndarray, AV: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    All diagnostics for one head. If the head output AV (n x d_head) is given,
    also returns its spectral entropy H(AV) and the normalised effective
    dimension (exp H - 1) / (d - 1) in [0, 1].
    """
    out = {name: fn(A) for name, fn in DIAGNOSTICS.items()}
    if AV is not None:
        h = spectral_entropy(AV)
        d = AV.shape[1]
        out["output_entropy"] = h
        out["output_dim"] = float(np.clip((np.exp(h) - 1.0) / max(d - 1.0, 1.0), 0.0, 1.0))
    return out
