"""Numerical checks of the statements in Sec. 3 of the paper."""

import numpy as np
import pytest

from spectral_attention import synthetic as syn
from spectral_attention.tdo import (
    intrinsic_measures,
    mean_routing_decomposition,
    spectral_entropy,
    tdo_pr,
    tdo_pr_centered,
    tdo_pr_from_traces,
    tdo_spectrum,
    token_difference_operator,
    weighted_svd,
)

RNG = np.random.default_rng(0)
MAPS = [syn.random_causal(24, temperature=t, sink_logit=s, rng=RNG)
        for t in (0.3, 1.0, 3.0) for s in (0.0, 3.0)]


@pytest.mark.parametrize("A", MAPS)
def test_euclidean_noncontractive(A):
    """Prop. 1: sigma_1(A) >= 1 for row-stochastic A."""
    assert np.linalg.svd(A, compute_uv=False)[0] >= 1 - 1e-12


def test_euclidean_equality_iff_doubly_stochastic():
    assert np.isclose(np.linalg.svd(syn.copy_map(10), compute_uv=False)[0], 1.0)
    # extreme causal sink: sigma_1^2 = n, all energy on the sink coordinate
    assert np.isclose(np.linalg.svd(syn.sink_map(10), compute_uv=False)[0] ** 2, 10.0)


@pytest.mark.parametrize("A", MAPS)
def test_intrinsic_non_expansion(A):
    """Prop. 2: ||Af||_pi <= ||f||_mu, lambda_1(D) = 1 attained by the constant."""
    mu, pi = intrinsic_measures(A)
    for f in RNG.standard_normal((50, A.shape[0])):
        assert pi @ (A @ f) ** 2 <= mu @ f ** 2 + 1e-12
    lam = tdo_spectrum(A)
    assert np.isclose(lam[0], 1.0)
    D = token_difference_operator(A)
    one = np.ones(A.shape[0])
    assert np.allclose(D @ one, one)
    assert np.allclose(mu @ D, mu)


@pytest.mark.parametrize("A", MAPS)
def test_centered_contraction(A):
    """Cor. 1: on mu-centered signals the gain is sqrt(lambda_2) <= 1."""
    mu, pi = intrinsic_measures(A)
    sigma, U, V = weighted_svd(A)
    assert np.isclose(sigma[0], 1.0)
    assert np.allclose(V[:, 0] * np.sign(V[0, 0]), 1.0)       # first right mode = constant
    f2 = V[:, 1]
    assert np.isclose(mu @ f2, 0.0, atol=1e-10)
    gain = np.sqrt(pi @ (A @ f2) ** 2 / (mu @ f2 ** 2))
    assert np.isclose(gain, np.sqrt(tdo_spectrum(A)[1]))
    for f in RNG.standard_normal((50, A.shape[0])):
        f = f - mu @ f
        assert np.sqrt(pi @ (A @ f) ** 2 / (mu @ f ** 2)) <= sigma[1] + 1e-10


@pytest.mark.parametrize("A", MAPS)
def test_mean_routing_decomposition(A):
    """Prop. 3 and Cor. 2."""
    n, d = A.shape[0], 8
    V = RNG.standard_normal((n, d))
    mu, pi = intrinsic_measures(A)
    m, V_c, AV_c = mean_routing_decomposition(A, V)
    assert np.allclose(A @ V, m[None, :] + AV_c)
    assert np.allclose(mu @ V_c, 0.0)
    assert np.isclose(np.sum(pi[:, None] * m[None, :] * AV_c), 0.0, atol=1e-10)   # pi-orthogonal
    lam2 = tdo_spectrum(A)[1]
    assert np.sum(pi[:, None] * AV_c ** 2) <= lam2 * np.sum(mu[:, None] * V_c ** 2) + 1e-10
    assert np.allclose(pi @ (A @ V), m)                                           # barycenters align

    Z = np.sqrt(pi)[:, None] * (A @ V)
    Z_c = np.sqrt(pi)[:, None] * AV_c
    alpha = np.sum(Z_c ** 2) / (m @ m + np.sum(Z_c ** 2))
    h2 = -alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha)
    H, H_c = spectral_entropy(Z), spectral_entropy(Z_c)
    assert alpha * H_c - 1e-10 <= H <= h2 + alpha * H_c + 1e-10


def test_pr_extremes():
    n = 16
    assert np.isclose(tdo_pr(syn.sink_map(n)), 1 / n)
    assert tdo_pr_centered(syn.sink_map(n)) == 0.0
    assert np.isclose(tdo_pr(syn.copy_map(n)), 1.0)
    assert np.isclose(tdo_pr(syn.shift_map(n)), (n - 1) / n)     # one key is never attended
    assert np.isclose(tdo_pr(syn.sink_map(n), "euclidean"), 1 / n)


@pytest.mark.parametrize("A", MAPS)
def test_traces_match_spectrum(A):
    assert np.isclose(tdo_pr_from_traces(A), tdo_pr(A))
    assert np.isclose(tdo_pr_from_traces(A, centered=True), tdo_pr_centered(A))


def test_sink_bias():
    """Mixing in a sink collapses the Euclidean PR much faster than the intrinsic one."""
    n = 32
    A = syn.interpolate(syn.copy_map(n), syn.sink_map(n), 0.5)
    assert tdo_pr(A, "euclidean") < 0.15
    assert tdo_pr(A, "intrinsic") > 0.9
