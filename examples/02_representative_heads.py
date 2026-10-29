"""
Representative heads of a real model and how each diagnostic scores them (Fig. 3).

Four routing patterns are picked automatically from one forward pass:
  sink         all queries collapse onto the first token (highest sink score)
  sink-biased  large sink mass coexisting with token-dependent routing
               (sink score > 0.5, largest gap intrinsic PR - Euclidean PR)
  shifting     previous-token head (highest A[i, i-1])
  copying      each token attends to itself (highest mean A[i, i])

    python examples/02_representative_heads.py --model gpt2
    python examples/02_representative_heads.py --model meta-llama/Llama-3.1-8B   # paper model
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _style as st
from _heads import head_metrics, load, load_texts
from spectral_attention.diagnostics import LABELS


def pick_heads(m):
    flat = lambda k: m[k].ravel()
    n_heads = m["sink_score"].shape[1]
    attn = m["_attention"]
    self_attn = np.array([[np.diag(A).mean() for A in layer] for layer in attn]).ravel()
    sink, intr, eucl = flat("sink_score"), flat("intrinsic_pr"), flat("euclidean_pr")

    biased = np.where(sink > 0.5, intr - eucl, -np.inf)
    picks = {
        "Copying": int(np.argmax(self_attn)),
        "Sink": int(np.argmax(sink)),
        "Sink-biased": int(np.argmax(biased)),
        "Shifting": int(np.argmax(flat("prev_token_score"))),
    }
    return {name: divmod(idx, n_heads) for name, idx in picks.items()}


def main(model_name: str, max_show: int):
    tok, model = load(model_name)
    m = head_metrics(model, tok, load_texts(1)[0])
    heads = pick_heads(m)

    metrics = list(LABELS) + ["output_dim"]
    labels = [LABELS.get(k, k) for k in metrics]
    labels[-1] = "Output dim. of AV"
    colors = [st.BLUE if k.startswith("intrinsic") else st.YELLOW if k == "output_dim" else st.BASELINE
              for k in metrics]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.25})
    for col, (name, (l, h)) in enumerate(heads.items()):
        A = m["_attention"][l][h][:max_show, :max_show]
        ax = axes[0, col]
        ax.imshow(A, vmin=0, vmax=1)
        ax.set_title(f"{name}  (L{l} H{h})")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

        vals = [m[k][l, h] for k in metrics]
        vals[metrics.index("log_det_score")] = 1 - np.exp(vals[metrics.index("log_det_score")])   # map to [0, 1]
        ax = axes[1, col]
        y = np.arange(len(metrics))[::-1]
        ax.barh(y, vals, color=colors, height=0.7)
        for yi, v in zip(y, vals):
            ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=7, color=st.INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if col == 0 else [], fontsize=8)
        ax.set_xlim(0, 1.2)
        ax.grid(axis="y", visible=False)
    axes[1, 0].annotate("log-det shown as 1 - exp(score)", (0, -0.2), xycoords="axes fraction",
                        fontsize=7, color=st.INK_2)
    fig.suptitle(f"Representative attention heads in {model_name}", y=0.97)
    st.save(fig, f"representative_heads_{model_name.split('/')[-1]}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--max-show", type=int, default=40)
    args = p.parse_args()
    main(args.model, args.max_show)
