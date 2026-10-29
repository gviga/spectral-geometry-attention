"""lm-polygraph estimator for SAUQ (sequence level, higher = more uncertain)."""

from typing import Dict

import numpy as np
from lm_polygraph.estimators.estimator import Estimator

from ..sauq import sauq_from_signals


class SAUQ(Estimator):
    """
    Spectral Attention Uncertainty Quantification.

    Consumes `sauq_signals` (from SAUQSignalCalculator) and the greedy token
    log-likelihoods, runs the RAUQ-style confidence recurrence per layer and
    returns the maximum layer uncertainty.

    Args:
        alpha: weight of the token probability in the recurrence (paper: 0.2).
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__(["sauq_signals", "greedy_log_likelihoods"], "sequence")
        self.alpha = alpha

    def __str__(self) -> str:
        return f"SAUQ(a={self.alpha})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            sauq_from_signals(np.asarray(logp), {l: d["signal"] for l, d in sig.items()}, self.alpha)
            for logp, sig in zip(stats["greedy_log_likelihoods"], stats["sauq_signals"])
        ])


def load_estimator(config):
    """Entry point used by `polygraph_eval` (`name: spectral_attention.lm_polygraph.estimator`)."""
    return SAUQ(**config)
