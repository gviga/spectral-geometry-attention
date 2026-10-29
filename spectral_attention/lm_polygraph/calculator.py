"""
lm-polygraph stat calculator producing the SAUQ routing signals.

It runs one teacher-forced forward pass per instance on prompt + greedy
generation and keeps only, per monitored layer, the selected head and its
per-step signal. Full attention tensors never leave the calculator, so the
default GreedyProbsCalculator can run with `output_attentions: false`.

Each instance is re-tokenised on its own, so the attention never contains
padding and never mixes instances of a batch.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator

from ..hf import attention_maps
from ..sauq import middle_layers, routing_signals


class SAUQSignalCalculator(StatCalculator):
    """
    Computes `sauq_signals`: per instance, a dict {layer: {"head", "signal"}}
    with `signal` of length T-1 (T = number of generated tokens).

    Args:
        variant: "intrinsic" (SAUQ, PR of the intrinsic TDO on the full
            prefix) or "generation" (PR on the generated-token block only).
        layers: layers to monitor; defaults to the middle third.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["sauq_signals"], ["greedy_tokens"]

    def __init__(self, variant: str = "intrinsic", layers: Optional[Sequence[int]] = None):
        super().__init__()
        self.variant = variant
        self.layers = None if layers is None else list(layers)

    def __call__(self, dependencies: Dict[str, np.ndarray], texts: List[str], model,
                 max_new_tokens: int = 100, **kwargs) -> Dict[str, list]:
        hf_model = model.model
        layers = self.layers or middle_layers(hf_model.config.num_hidden_layers)
        signals = []
        for text, gen_ids in zip(texts, dependencies["greedy_tokens"]):
            prompt_ids = model.tokenize([text])["input_ids"][0].tolist()
            attn = attention_maps(hf_model, prompt_ids + list(gen_ids), layers)
            attentions = [attn.get(l) for l in range(max(layers) + 1)]
            signals.append(routing_signals(attentions, len(prompt_ids), len(gen_ids), layers, self.variant))
        return {"sauq_signals": signals}


def load_stat_calculator(config, environment):
    """Entry point used by `polygraph_eval` (`builder: spectral_attention.lm_polygraph.calculator`)."""
    return SAUQSignalCalculator(
        variant=config.get("variant", "intrinsic"),
        layers=config.get("layers", None),
    )
