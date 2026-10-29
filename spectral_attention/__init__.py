"""Spectral Geometry of Attention: Token Difference Operator and SAUQ."""

from .tdo import (
    intrinsic_measures,
    counting_measures,
    token_difference_operator,
    tdo_symmetric,
    tdo_spectrum,
    weighted_svd,
    participation_ratio,
    tdo_pr,
    tdo_pr_centered,
    tdo_pr_from_traces,
    spectral_entropy,
    mean_routing_decomposition,
)
from .diagnostics import head_diagnostics, DIAGNOSTICS
from .sauq import (
    middle_layers,
    selection_scores,
    intrinsic_signal,
    generation_signal,
    confidence_trajectory,
    routing_signals,
    sauq_score,
)

__version__ = "0.1.0"
