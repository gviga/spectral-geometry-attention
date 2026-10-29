"""
Token Difference Operator (TDO) of an attention map.

An attention head A in R^{n x n} (row-stochastic, causal) is viewed as a linear
operator between weighted Hilbert spaces of functions on the token positions,

    A : L^2(mu) -> L^2(pi),        f  |->  A f,

with adjoint A* = Omega_mu^{-1} A^T Omega_pi. The Token Difference Operator is

    D_{mu,pi} = A* A = Omega_mu^{-1} A^T Omega_pi A,

self-adjoint and PSD in L^2(mu). Its spectrum is computed from the symmetric,
similar matrix

    S_{mu,pi} = Omega_mu^{-1/2} A^T Omega_pi A Omega_mu^{-1/2} = B^T B,
    B         = Omega_pi^{1/2} A Omega_mu^{-1/2}.

Two geometries are studied in the paper (Sec. 3):

* "euclidean":  mu = pi = 1 (counting measure), D = A^T A.
  Biased by sinks: sigma_1(A) >= 1, and the constant mode is mapped to the
  column-sum profile (Prop. 1).
* "intrinsic":  pi = 1/n,  mu = A^T pi (the average attention row).
  Non-expansive with lambda_1(D) = 1 attained by the constant function (Prop. 2);
  token-dependent routing lives on the mu-centered subspace (Cor. 1).

Keys with mu_j = 0 (never attended to) carry no mass in L^2(mu): they are
dropped from the spectrum, which is the natural restriction to supp(mu).
"""

from typing import Optional, Tuple, Union

import numpy as np

_EPS = 1e-12

Geometry = Union[str, Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def intrinsic_measures(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu, pi) with pi = 1/n uniform and mu = A^T pi (Eq. 8)."""
    n = A.shape[0]
    pi = np.full(n, 1.0 / n)
    mu = A.T @ pi
    return mu, pi


def counting_measures(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu, pi) = (1, 1), the Euclidean / counting geometry."""
    n_rows, n_cols = A.shape
    return np.ones(n_cols), np.ones(n_rows)


def resolve_measures(A: np.ndarray, geometry: Geometry = "intrinsic") -> Tuple[np.ndarray, np.ndarray]:
    """Map a geometry name, or an explicit (mu, pi) pair, to measure vectors."""
    if isinstance(geometry, tuple):
        mu, pi = geometry
        return np.asarray(mu, dtype=float), np.asarray(pi, dtype=float)
    if geometry == "intrinsic":
        return intrinsic_measures(A)
    if geometry == "euclidean":
        return counting_measures(A)
    raise ValueError(f"geometry must be 'intrinsic', 'euclidean' or (mu, pi), got {geometry!r}")


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def embedded_operator(A: np.ndarray, mu: np.ndarray, pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isometric Euclidean embedding B = Omega_pi^{1/2} A Omega_mu^{-1/2}.

    Returns (B, support) where `support` is the boolean mask of keys with
    mu_j > 0; B only contains those columns.
    """
    support = mu > _EPS
    B = np.sqrt(pi)[:, None] * A[:, support] / np.sqrt(mu[support])[None, :]
    return B, support


def token_difference_operator(A: np.ndarray, geometry: Geometry = "intrinsic") -> np.ndarray:
    """
    D_{mu,pi} = Omega_mu^{-1} A^T Omega_pi A  (Eq. 6), as an n x n matrix.

    Rows of keys outside supp(mu) are set to zero.
    """
    mu, pi = resolve_measures(A, geometry)
    inv_mu = np.where(mu > _EPS, 1.0 / np.maximum(mu, _EPS), 0.0)
    return inv_mu[:, None] * (A.T @ (pi[:, None] * A))


def tdo_symmetric(A: np.ndarray, geometry: Geometry = "intrinsic") -> np.ndarray:
    """S_{mu,pi} = B^T B, symmetric and similar to D_{mu,pi} on supp(mu)."""
    mu, pi = resolve_measures(A, geometry)
    B, _ = embedded_operator(A, mu, pi)
    return B.T @ B


def tdo_spectrum(A: np.ndarray, geometry: Geometry = "intrinsic") -> np.ndarray:
    """Eigenvalues of D_{mu,pi} in descending order (squared singular values of B)."""
    mu, pi = resolve_measures(A, geometry)
    B, _ = embedded_operator(A, mu, pi)
    sigma = np.linalg.svd(B, compute_uv=False)
    return sigma ** 2


def weighted_svd(A: np.ndarray, geometry: Geometry = "intrinsic") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD of A : L^2(mu) -> L^2(pi).

    Returns (sigma, U, V) with sigma descending, U pi-orthonormal
    (U^T Omega_pi U = I) and V mu-orthonormal (V^T Omega_mu V = I), such that
    A = U diag(sigma) V^T Omega_mu. In the intrinsic geometry sigma_1 = 1 and
    the first right singular vector is the constant function.
    """
    mu, pi = resolve_measures(A, geometry)
    B, support = embedded_operator(A, mu, pi)
    U_b, sigma, Vt_b = np.linalg.svd(B, full_matrices=False)
    U = U_b / np.sqrt(pi)[:, None]
    V = np.zeros((A.shape[1], len(sigma)))
    V[support] = Vt_b.T / np.sqrt(mu[support])[:, None]
    return sigma, U, V


# ---------------------------------------------------------------------------
# Spectral summaries
# ---------------------------------------------------------------------------

def participation_ratio(eigenvalues: np.ndarray, n: Optional[int] = None) -> float:
    """
    Normalised participation ratio (Eq. 13):  (1/n) (sum lambda)^2 / sum lambda^2.

    Ranges from 1/n (one active mode) to rank/n (flat spectrum). If `n` is
    None it defaults to len(eigenvalues).
    """
    lam = np.asarray(eigenvalues, dtype=float)
    n = len(lam) if n is None else n
    denom = np.sum(lam ** 2)
    if n == 0 or denom <= _EPS:
        return 0.0
    return float(lam.sum() ** 2 / denom / n)


def tdo_pr(A: np.ndarray, geometry: Geometry = "intrinsic") -> float:
    """PR(D_{mu,pi}) normalised by the sequence length n."""
    return participation_ratio(tdo_spectrum(A, geometry), n=A.shape[0])


def tdo_pr_centered(A: np.ndarray) -> float:
    """
    Centered PR in the intrinsic geometry:  (1/(n-1)) (sum_{k>=2} lambda_k)^2 / sum_{k>=2} lambda_k^2.

    Drops the trivial unit mode lambda_1 = 1 (the constant function); returns
    0 when no centered mode survives (e.g. a pure sink).
    """
    n = A.shape[0]
    if n < 2:
        return 0.0
    lam = tdo_spectrum(A, "intrinsic")[1:]
    return participation_ratio(lam, n=n - 1)


def tdo_pr_from_traces(A: np.ndarray, centered: bool = False) -> float:
    """
    Eigendecomposition-free intrinsic PR, from tr(S) and ||S||_F^2 only.

    This is how SAUQ computes its per-step signal. With intrinsic measures the
    top eigenvalue is exactly 1, so the centered PR follows by subtracting it
    from both traces.
    """
    mu, pi = intrinsic_measures(A)
    B, _ = embedded_operator(A, mu, pi)
    G = B @ B.T                      # shares the nonzero spectrum of S = B^T B
    tr, fro2 = np.trace(G), np.sum(G * G)
    n = A.shape[0]
    if centered:
        tr, fro2, n = tr - 1.0, fro2 - 1.0, n - 1
    if n <= 0 or fro2 <= _EPS or tr <= _EPS:
        return 0.0
    return float(tr * tr / fro2 / n)


def spectral_entropy(Y: np.ndarray) -> float:
    """H(Y) = -sum p_k log p_k with p_k = sigma_k(Y)^2 / ||Y||_F^2."""
    s2 = np.linalg.svd(Y, compute_uv=False) ** 2
    total = s2.sum()
    if total <= _EPS:
        return 0.0
    p = s2[s2 > _EPS * total] / total
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# From functions to representations (Sec. 3.2)
# ---------------------------------------------------------------------------

def mean_routing_decomposition(A: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the head output as AV = 1 m + A V_c (Prop. 3).

    Returns (m, V_c, AV_c) with m = mu^T V the attention-weighted mean value
    and V_c = V - 1 m its mu-centered part. The routed term satisfies
    ||A V_c||_{pi,F} <= sqrt(lambda_2(D)) ||V_c||_{mu,F}.
    """
    mu, _ = intrinsic_measures(A)
    m = mu @ V
    V_c = V - m[None, :]
    return m, V_c, A @ V_c
