"""
Hugging Face helpers: attention maps, per-head outputs, and SAUQ end to end.

Models must be loaded with `attn_implementation="eager"`; fused attention
kernels (sdpa, flash) do not return attention weights.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .sauq import middle_layers, routing_signals, layer_uncertainty


def _as_ids(input_ids, device) -> torch.Tensor:
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    return (ids[None] if ids.ndim == 1 else ids).to(device)


@torch.no_grad()
def attention_maps(model, input_ids, layers: Optional[Sequence[int]] = None) -> Dict[int, np.ndarray]:
    """Teacher-forced forward pass; returns {layer: (H, N, N) float64 attention}."""
    out = model(_as_ids(input_ids, model.device), output_attentions=True)
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError("No attention weights returned: load the model with attn_implementation='eager'.")
    layers = range(len(out.attentions)) if layers is None else layers
    return {l: out.attentions[l][0].float().cpu().numpy().astype(np.float64) for l in layers}


# Input of the attention output projection = concatenation of per-head A_h V_h.
_OUT_PROJ_SUFFIXES = (
    ".attn.c_proj",            # GPT-2
    ".self_attn.o_proj",       # Llama, Mistral, Qwen, Gemma
    ".attention.dense",        # GPT-NeoX / Pythia
    ".self_attention.dense",   # BLOOM, Falcon
)


@torch.no_grad()
def attention_and_head_outputs(model, input_ids) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    One forward pass returning, per layer, the attention maps (H, N, N) and the
    head outputs A_h V_h (H, N, d_head), captured before the output projection.
    """
    n_heads = model.config.num_attention_heads
    modules = [m for name, m in model.named_modules() if name.endswith(_OUT_PROJ_SUFFIXES)]
    if len(modules) != model.config.num_hidden_layers:
        raise RuntimeError("Could not locate the attention output projections for this architecture.")

    captured: List[torch.Tensor] = []
    hooks = [m.register_forward_pre_hook(lambda _m, args: captured.append(args[0][0].detach()))
             for m in modules]
    try:
        out = model(_as_ids(input_ids, model.device), output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    attn = [a[0].float().cpu().numpy().astype(np.float64) for a in out.attentions]
    head_out = []
    for x in captured:
        n, hidden = x.shape
        head_out.append(x.float().cpu().numpy().reshape(n, n_heads, hidden // n_heads).transpose(1, 0, 2))
    return attn, head_out


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 64,
                    chat: bool = False) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Greedy decoding. Returns (prompt_ids, generated_ids, token_logprobs); the
    EOS token, if produced, is kept as the last generated token.
    """
    if chat:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, return_tensors="pt")
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    ids = ids.to(model.device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, output_scores=True, return_dict_in_generate=True,
                         pad_token_id=pad)
    gen = out.sequences[0, ids.shape[1]:].tolist()
    if tokenizer.eos_token_id in gen:
        gen = gen[:gen.index(tokenizer.eos_token_id) + 1]
    logprobs = np.array([torch.log_softmax(out.scores[t][0].float(), -1)[tok].item()
                         for t, tok in enumerate(gen)])
    return ids[0].tolist(), gen, logprobs


def sauq_for_generation(
    model,
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    token_logprobs: np.ndarray,
    alpha: float = 0.2,
    layers: Optional[Sequence[int]] = None,
    variant: str = "intrinsic",
) -> Tuple[float, dict]:
    """
    SAUQ for one (prompt, generation) pair from a single extra forward pass.

    Returns (uncertainty, details) with details[layer] = {"head", "signal", "uncertainty"}.
    """
    layers = middle_layers(model.config.num_hidden_layers) if layers is None else list(layers)
    attn = attention_maps(model, list(prompt_ids) + list(generated_ids), layers)
    attentions = [attn.get(l) for l in range(max(layers) + 1)]
    details = routing_signals(attentions, len(prompt_ids), len(generated_ids), layers, variant)
    for d in details.values():
        d["uncertainty"] = layer_uncertainty(token_logprobs, d["signal"], alpha)
    return max(d["uncertainty"] for d in details.values()), details
