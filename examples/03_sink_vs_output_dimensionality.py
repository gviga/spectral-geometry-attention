"""
Sink dependence and output dimensionality across all heads of a model (Fig. 4).

For every head, each diagnostic is averaged over a few WikiText passages and
plotted against (top) the sink score and (bottom) the spectral entropy H(AV)
of the head output. A good routing diagnostic should depend little on the
sink score and track H(AV); the Spearman correlation is shown in each panel.

    python examples/03_sink_vs_output_dimensionality.py --model gpt2 --n-texts 20
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import _style as st
from _heads import head_metrics, load, load_texts
from spectral_attention.diagnostics import LABELS

METRICS = [k for k in LABELS if k != "sink_score"]


def main(model_name: str, n_texts: int, max_len: int):
    tok, model = load(model_name)
    runs = [head_metrics(model, tok, t, max_len) for t in load_texts(n_texts)]
    mean = {k: np.mean([r[k] for r in runs], axis=0).ravel() for k in METRICS + ["sink_score", "output_entropy"]}

    rows = [("sink_score", "sink score"), ("output_entropy", "H(AV)")]
    fig, axes = plt.subplots(2, len(METRICS), figsize=(2.1 * len(METRICS), 4.6), sharex="row")
    for r, (x_key, x_label) in enumerate(rows):
        for c, k in enumerate(METRICS):
            ax = axes[r, c]
            color = st.BLUE if k.startswith("intrinsic") else st.INK_2
            ax.scatter(mean[x_key], mean[k], s=7, alpha=0.6, color=color, linewidths=0)
            rho = spearmanr(mean[x_key], mean[k])[0]
            ax.set_title(rf"$\rho$ = {rho:+.2f}", fontsize=8.5, color=color)
            ax.set_xlabel(x_label, fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.text(0.5, 1.28, LABELS[k], transform=ax.transAxes, ha="center", fontsize=8.5)
    fig.suptitle(f"{model_name}: {mean['sink_score'].size} heads, averaged over {len(runs)} passages "
                 "(top: vs sink score, bottom: vs output entropy)", y=1.04, fontsize=10)
    fig.tight_layout()
    st.save(fig, f"sink_vs_output_{model_name.split('/')[-1]}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--n-texts", type=int, default=20)
    p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args()
    main(args.model, args.n_texts, args.max_len)
