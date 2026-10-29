"""Shared helpers for the real-model examples: load a model, texts, per-head metrics."""

from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectral_attention.diagnostics import head_diagnostics
from spectral_attention.hf import attention_and_head_outputs

FALLBACK_TEXTS = [
    "The European Space Agency launched a new satellite last Thursday to study climate patterns across "
    "the northern hemisphere. Scientists from twelve countries will analyse the data over the next three "
    "years, hoping to improve forecasts of droughts, floods and heat waves.",
    "In 1859 Charles Darwin published On the Origin of Species, introducing the theory that populations "
    "evolve over the course of generations through a process of natural selection. The book was "
    "controversial because it contradicted religious beliefs that underlay the theories of biology.",
    "The recipe calls for two cups of flour, one cup of sugar, three eggs and a pinch of salt. Mix the dry "
    "ingredients first, then add the eggs one at a time, stirring until the batter is smooth.",
    "After the match, the coach said the team had played its best football of the season, but admitted "
    "that the defence still needed work before the final against their old rivals next month.",
]


def load(model_name: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    return tok, model


def load_texts(n: int, min_words: int = 60) -> List[str]:
    """First n WikiText-103 test paragraphs (falls back to a few built-in texts)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
        texts = [t.strip() for t in ds["text"] if len(t.split()) >= min_words and not t.strip().startswith("=")]
        return texts[:n]
    except Exception as e:                          # offline or `datasets` missing
        print(f"WikiText unavailable ({type(e).__name__}); using built-in texts")
        return FALLBACK_TEXTS[:n]


def head_metrics(model, tok, text: str, max_len: int = 128) -> Dict[str, np.ndarray]:
    """Diagnostics for every head on one text; returns {metric: (L, H) array} plus attention maps."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
    attn, head_out = attention_and_head_outputs(model, ids)
    per_head = [[head_diagnostics(A, AV) for A, AV in zip(attn[l], head_out[l])] for l in range(len(attn))]
    keys = per_head[0][0].keys()
    out = {k: np.array([[h[k] for h in layer] for layer in per_head]) for k in keys}
    out["_attention"] = attn
    out["_tokens"] = tok.convert_ids_to_tokens(ids)
    return out
