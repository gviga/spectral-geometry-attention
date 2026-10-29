"""
Euclidean vs intrinsic geometry on synthetic attention (Sec. 3, teaser center).

We interpolate A(alpha) = (1 - alpha) I + alpha 1 e_1^T between a copy map and
a sink map. The Euclidean PR(A^T A) collapses as soon as a little sink mass is
mixed in, although every token is still routed to itself; the intrinsic
PR(D_{mu,pi}) only collapses when routing actually does (alpha -> 1).

    python examples/01_sink_bias.py --n 32
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _style as st
from spectral_attention import synthetic as syn
from spectral_attention.diagnostics import sink_score
from spectral_attention.tdo import tdo_pr, tdo_pr_centered, tdo_spectrum


def main(n: int, alpha_ref: float):
    alphas = np.linspace(0, 1, 101)
    maps = [syn.interpolate(syn.copy_map(n), syn.sink_map(n), a) for a in alphas]
    curves = {
        r"PR($A^\top A$)  Euclidean": ([tdo_pr(A, "euclidean") for A in maps], st.ORANGE),
        r"PR($D_{\mu,\pi}$)  intrinsic": ([tdo_pr(A) for A in maps], st.BLUE),
        r"PR$_\mathrm{cent}$($D_{\mu,\pi}$)": ([tdo_pr_centered(A) for A in maps], st.AQUA),
    }

    fig = plt.figure(figsize=(12, 3.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 2.3, 1.9], wspace=0.45)

    for k, a in enumerate([0.0, alpha_ref, 1.0]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(syn.interpolate(syn.copy_map(n), syn.sink_map(n), a), vmin=0, vmax=1)
        ax.set_title(rf"$\alpha$ = {a:g}")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.axes[0].set_ylabel("query")

    ax = fig.add_subplot(gs[0, 3])
    for label, (vals, color) in curves.items():
        ax.plot(alphas, vals, color=color, label=label)
    ax.plot(alphas, [sink_score(A) for A in maps], color=st.BASELINE, ls="--", lw=1.5, label="sink score")
    ax.axvline(alpha_ref, color=st.INK_2, lw=0.8, ls=":")
    ax.set_xlabel(r"sink weight $\alpha$  (copy $\to$ sink)")
    ax.set_ylabel("normalised score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Participation ratio along the interpolation")

    ax = fig.add_subplot(gs[0, 4])
    A = syn.interpolate(syn.copy_map(n), syn.sink_map(n), alpha_ref)
    k = np.arange(1, n + 1)
    ax.semilogy(k, tdo_spectrum(A, "euclidean"), "o-", color=st.ORANGE, ms=3, lw=1.2, label="Euclidean")
    ax.semilogy(k, tdo_spectrum(A, "intrinsic"), "o-", color=st.BLUE, ms=3, lw=1.2, label="intrinsic")
    ax.set_xlabel("mode k")
    ax.set_ylabel(r"eigenvalue $\lambda_k$")
    ax.set_title(rf"Spectrum at $\alpha$ = {alpha_ref:g}")
    ax.legend(fontsize=7.5)

    st.save(fig, "sink_bias.png")
    print(f"alpha={alpha_ref}:  Euclidean PR={tdo_pr(A, 'euclidean'):.3f}  intrinsic PR={tdo_pr(A):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()
    main(args.n, args.alpha)
