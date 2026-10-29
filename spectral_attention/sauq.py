"""
SAUQ: Spectral Attention Uncertainty Quantification (Sec. 5).

Notation. The prompt has P tokens and the model generates T tokens
y_0 .. y_{T-1}. A teacher-forced forward pass on the N = P + T tokens gives,
for every layer, a causal attention tensor of shape (H, N, N). Position
P + t - 1 is the query that predicts y_t (position P - 1 predicts y_0).

For every generation step t = 1 .. T-1 and every layer in the middle third:

1. Head selection (Eq. 14). On the generated-token block of the final step,
       B = A[P-1 : P+T-1, P : P+T]     (rows: the queries that produced y_0..y_{T-1};
                                         cols: the generated keys),
   pick h* = argmax_h PR(B_h B_h^T). One batched product over all heads.

2. Routing signal (Eq. 15). On the selected head, the prefix seen when
   predicting y_t is A_t = A[:P+t, :P+t]. Equip it with the intrinsic
   measures pi = 1/n, mu = A_t^T pi and take
       s_t = PR(D_{mu,pi}(A_t)).
   With S = C^{-1/2} W C^{-1/2}, W = A_t^T A_t and C = diag(column sums),
   tr(S) and ||S||_F^2 are updated in O(n^2) per step and no
   eigendecomposition is needed.

   "generation" variant: s_t = PR(B_t B_t^T) / (t+1) on the growing generated
   block B_t = A[P-1 : P+t, P : P+t+1] (sub-stochastic; no intrinsic geometry).

3. Confidence recurrence (Eq. 16), as in RAUQ:
       c_0 = p_0,    c_t = alpha p_t + (1 - alpha) s_t c_{t-1},
   with p_t the probability of the generated token y_t.

4. U^{(l)} = 1 - mean_t log c_t, and U_SAUQ = max over middle layers.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np

_EPS = 1e-10


def middle_layers(n_layers: int) -> list:
    """Middle third of the layers, as in RAUQ (layers 10-22 for a 32-layer model)."""
    return list(range(n_layers // 3, int(np.ceil(n_layers / 3 * 2)) + 1))


# ---------------------------------------------------------------------------
# Head selection
# ---------------------------------------------------------------------------

def generated_block(A: np.ndarray, prompt_len: int, n_steps: int) -> np.ndarray:
    """B = A[..., P-1 : P+T-1, P : P+T]: generated-key attention of the rows that produced y_0..y_{T-1}."""
    P, T = prompt_len, n_steps
    return A[..., P - 1:P + T - 1, P:P + T]


def selection_scores(A_layer: np.ndarray, prompt_len: int, n_steps: int) -> np.ndarray:
    """Un-normalised PR(B_h B_h^T) for every head of a layer; A_layer has shape (H, N, N)."""
    B = generated_block(A_layer, prompt_len, n_steps).astype(np.float64)
    G = B @ B.transpose(0, 2, 1)
    tr = np.einsum("hii->h", G)
    fro2 = np.einsum("hij,hij->h", G, G)
    return tr * tr / (fro2 + _EPS)


# ---------------------------------------------------------------------------
# Per-step routing signals
# ---------------------------------------------------------------------------

def intrinsic_signal(A: np.ndarray, prompt_len: int, n_steps: int) -> np.ndarray:
    """
    s_t = PR(D_{mu,pi}(A[:P+t, :P+t])) for t = 1 .. T-1, shape (T-1,).

    Incremental: adding query row a updates W += a a^T and c += a, after which
        tr(S)     = sum_j W_jj / c_j,
        ||S||_F^2 = sum_jk W_jk^2 / (c_j c_k).
    Keys that received no attention (c_j = 0) are outside supp(mu) and drop out.
    """
    P, T = prompt_len, n_steps
    n_max = P + T - 1
    A = np.asarray(A[:n_max, :n_max], dtype=np.float64)

    W = np.zeros((n_max, n_max))
    W[:P, :P] = A[:P, :P].T @ A[:P, :P]
    c = np.zeros(n_max)
    c[:P] = A[:P, :P].sum(axis=0)

    signal = np.empty(T - 1)
    for t in range(1, T):
        n = P + t
        a = A[n - 1, :n]
        W[:n, :n] += np.outer(a, a)
        c[:n] += a
        inv_c = np.where(c[:n] > _EPS, 1.0 / np.maximum(c[:n], _EPS), 0.0)
        Wn = W[:n, :n]
        tr = np.dot(np.diag(Wn), inv_c)
        fro2 = inv_c @ (Wn * Wn) @ inv_c
        signal[t - 1] = tr * tr / (fro2 + _EPS) / n
    return signal


def generation_signal(A: np.ndarray, prompt_len: int, n_steps: int) -> np.ndarray:
    """
    s_t = PR(B_t B_t^T) / (t+1) with B_t = A[P-1 : P+t, P : P+t+1], shape (T-1,).

    Bordered update of the row Gram: a new row b adds ||b||^2 to the trace and
    2 ||B_prev b||^2 + ||b||^4 to the squared Frobenius norm.
    """
    B = np.asarray(generated_block(A, prompt_len, n_steps), dtype=np.float64)
    T = n_steps
    tr = fro2 = B[0, 0] ** 2          # row of y_0's predictor: zero over generated keys
    signal = np.empty(T - 1)
    for t in range(1, T):
        b = B[t, :t + 1]
        cross = B[:t, :t + 1] @ b
        sq = np.dot(b, b)
        tr += sq
        fro2 += 2.0 * np.dot(cross, cross) + sq * sq
        signal[t - 1] = tr * tr / (fro2 + _EPS) / (t + 1)
    return signal


SIGNALS = {"intrinsic": intrinsic_signal, "generation": generation_signal}


# ---------------------------------------------------------------------------
# Recurrence and score
# ---------------------------------------------------------------------------

def confidence_trajectory(token_logprobs: np.ndarray, signal: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """c_0 = p_0,  c_t = alpha p_t + (1 - alpha) s_t c_{t-1}  (Eq. 16)."""
    p = np.exp(np.asarray(token_logprobs, dtype=np.float64))
    c = np.empty_like(p)
    c[0] = p[0]
    for t in range(1, len(p)):
        c[t] = alpha * p[t] + (1.0 - alpha) * signal[t - 1] * c[t - 1]
    return c


def layer_uncertainty(token_logprobs: np.ndarray, signal: np.ndarray, alpha: float = 0.2) -> float:
    """U^{(l)} = 1 - (1/T) sum_t log c_t."""
    c = confidence_trajectory(token_logprobs, signal, alpha)
    return float(1.0 - np.log(np.clip(c, _EPS, None)).mean())


def sauq_from_signals(token_logprobs: np.ndarray, signals: Dict[int, np.ndarray], alpha: float = 0.2) -> float:
    """Max over layers of the per-layer uncertainty; `signals` maps layer -> (T-1,) signal."""
    return max(layer_uncertainty(token_logprobs, s, alpha) for s in signals.values())


def routing_signals(
    attentions: Sequence[np.ndarray],
    prompt_len: int,
    n_steps: int,
    layers: Optional[Sequence[int]] = None,
    variant: str = "intrinsic",
) -> Dict[int, dict]:
    """
    Head selection + per-step signal for every monitored layer.

    attentions: per-layer arrays of shape (H, N, N) with N >= P + T - 1 (+1 for
    the "generation" variant), e.g. the `attentions` of a Hugging Face forward
    pass on prompt + generated tokens. Returns {layer: {"head", "signal"}}.
    """
    if prompt_len < 1:
        raise ValueError("SAUQ needs a non-empty prompt")
    signal_fn = SIGNALS[variant]
    layers = middle_layers(len(attentions)) if layers is None else layers
    out = {}
    for layer in layers:
        A_layer = attentions[layer]
        head = int(np.argmax(selection_scores(A_layer, prompt_len, n_steps)))
        out[layer] = {"head": head, "signal": signal_fn(A_layer[head], prompt_len, n_steps)}
    return out


def sauq_score(
    attentions: Sequence[np.ndarray],
    prompt_len: int,
    token_logprobs: Union[Sequence[float], np.ndarray],
    alpha: float = 0.2,
    layers: Optional[Sequence[int]] = None,
    variant: str = "intrinsic",
    return_details: bool = False,
):
    """
    SAUQ uncertainty of one generation (higher = more uncertain).

    token_logprobs: log p(y_t | y_<t, x) of the T generated tokens.
    """
    token_logprobs = np.asarray(token_logprobs, dtype=np.float64)
    T = len(token_logprobs)
    details = routing_signals(attentions, prompt_len, T, layers, variant)
    for d in details.values():
        d["uncertainty"] = layer_uncertainty(token_logprobs, d["signal"], alpha)
    score = max(d["uncertainty"] for d in details.values())
    return (score, details) if return_details else score
