"""The fast SAUQ paths agree with direct computations on each prefix."""

import numpy as np

from spectral_attention import synthetic as syn
from spectral_attention.sauq import (
    confidence_trajectory,
    generation_signal,
    intrinsic_signal,
    middle_layers,
    sauq_score,
    selection_scores,
)
from spectral_attention.tdo import tdo_pr

RNG = np.random.default_rng(1)
P, T, H, L = 9, 14, 4, 6
N = P + T


def _layers():
    return [np.stack([syn.random_causal(N, temperature=t, sink_logit=s, rng=RNG)
                      for t, s in zip((0.3, 1.0, 2.0, 0.5), (0.0, 2.0, 4.0, 1.0))])
            for _ in range(L)]


def _pr_rows(B):
    G = B @ B.T
    return np.trace(G) ** 2 / np.sum(G * G)


def test_intrinsic_signal_matches_direct():
    A = syn.random_causal(N, sink_logit=2.0, rng=RNG)
    direct = [tdo_pr(A[:P + t, :P + t]) for t in range(1, T)]
    assert np.allclose(intrinsic_signal(A, P, T), direct)


def test_intrinsic_signal_handles_unattended_keys():
    A = syn.interpolate(syn.shift_map(N), syn.sink_map(N), 0.3)   # key N-1 never attended
    direct = [tdo_pr(A[:P + t, :P + t]) for t in range(1, T)]
    assert np.allclose(intrinsic_signal(A, P, T), direct)


def test_generation_signal_matches_direct():
    A = syn.random_causal(N, rng=RNG)
    direct = [_pr_rows(A[P - 1:P + t, P:P + t + 1]) / (t + 1) for t in range(1, T)]
    assert np.allclose(generation_signal(A, P, T), direct)


def test_selection_matches_direct():
    A_layer = _layers()[0]
    direct = [_pr_rows(A_layer[h, P - 1:P + T - 1, P:P + T]) for h in range(H)]
    assert np.allclose(selection_scores(A_layer, P, T), direct)


def test_recurrence():
    logp = np.log([0.9, 0.5, 0.8])
    c = confidence_trajectory(logp, np.array([1.0, 0.5]), alpha=0.2)
    assert np.allclose(c, [0.9, 0.2 * 0.5 + 0.8 * 0.9, 0.2 * 0.8 + 0.8 * 0.5 * (0.1 + 0.72)])


def test_score_is_max_over_middle_layers():
    attentions = _layers()
    logp = np.log(RNG.uniform(0.2, 1.0, T))
    score, details = sauq_score(attentions, P, logp, return_details=True)
    assert sorted(details) == middle_layers(L)
    assert np.isclose(score, max(d["uncertainty"] for d in details.values()))
    # collapsed routing everywhere -> confidence is not propagated -> higher uncertainty
    sink = [np.stack([syn.sink_map(N)] * H)] * L
    assert sauq_score(sink, P, logp) > score
