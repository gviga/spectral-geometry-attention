"""
SAUQ on a Hugging Face model, without LM-Polygraph.

Answers a few trivia questions greedily, scores each answer with SAUQ and with
two probability baselines, and plots the routing signal s_t and the confidence
trajectory c_t of the most and least certain answers.

    python examples/04_sauq_quickstart.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _style as st
from _heads import load
from spectral_attention.hf import greedy_generate, sauq_for_generation
from spectral_attention.sauq import confidence_trajectory

QUESTIONS = [
    ("What is the capital of France?", "paris"),
    ("Which planet is known as the Red Planet?", "mars"),
    ("Who wrote the play Romeo and Juliet?", "shakespeare"),
    ("What is the chemical symbol for gold?", "au"),
    ("In which year did the Berlin Wall fall?", "1989"),
    ("Who painted the ceiling of the Sistine Chapel?", "michelangelo"),
    ("What is the smallest prime number?", "2"),
    ("Which element has atomic number 74?", "tungsten"),
    ("Who was the first woman to win a Nobel Prize?", "curie"),
    ("What is the capital of Burkina Faso?", "ouagadougou"),
    ("Which composer wrote the opera The Magic Flute?", "mozart"),
    ("What is the longest river in South America?", "amazon"),
    ("Who discovered penicillin?", "fleming"),
    ("Which country hosted the 1954 FIFA World Cup?", "switzerland"),
    ("What is the name of the largest moon of Saturn?", "titan"),
    ("Who formulated the uncertainty principle?", "heisenberg"),
]


def main(model_name: str, alpha: float, max_new_tokens: int):
    tok, model = load(model_name)
    chat = tok.chat_template is not None
    rows = []
    for q, gold in QUESTIONS:
        prompt = f"Answer with a few words only. {q}" if chat else f"Q: {q}\nA:"
        p_ids, g_ids, logp = greedy_generate(model, tok, prompt, max_new_tokens, chat=chat)
        if len(g_ids) < 2:
            continue
        u, details = sauq_for_generation(model, p_ids, g_ids, logp, alpha=alpha)
        answer = tok.decode(g_ids, skip_special_tokens=True).strip().split("\n")[0]
        rows.append(dict(q=q, answer=answer, correct=gold in answer.lower(), sauq=u,
                         msp=-logp.sum(), ppl=-logp.mean(), logp=logp, details=details))

    print(f"\n{'SAUQ':>6} {'-logP':>6} {'PPL':>5}  ok  answer")
    for r in sorted(rows, key=lambda r: r["sauq"]):
        print(f"{r['sauq']:6.2f} {r['msp']:6.2f} {r['ppl']:5.2f}  {'Y' if r['correct'] else '-'}   "
              f"{r['q']} -> {r['answer'][:50]}")
    ok = np.array([r["correct"] for r in rows])
    if 0 < ok.sum() < len(ok):
        from sklearn.metrics import roc_auc_score
        for key in ("sauq", "msp", "ppl"):
            print(f"AUROC(error | {key}) = {roc_auc_score(~ok, [r[key] for r in rows]):.3f}")

    lo, hi = min(rows, key=lambda r: r["sauq"]), max(rows, key=lambda r: r["sauq"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for r, color, name in [(lo, st.BLUE, "most certain"), (hi, st.ORANGE, "least certain")]:
        layer = max(r["details"], key=lambda l: r["details"][l]["uncertainty"])
        s = r["details"][layer]["signal"]
        c = confidence_trajectory(r["logp"], s, alpha)
        label = f"{name}: {r['answer'][:28]!r}"
        axes[0].plot(np.arange(1, len(s) + 1), s, "o-", ms=4, color=color, label=label)
        axes[1].semilogy(np.arange(len(c)), c, "o-", ms=4, color=color, label=label)
    axes[0].set(xlabel="generation step t", ylabel=r"routing signal $s_t$", title="Intrinsic PR of the selected head")
    axes[1].set(xlabel="generation step t", ylabel=r"confidence $c_t$", title="SAUQ confidence recurrence")
    axes[1].legend(fontsize=7.5)
    fig.tight_layout()
    st.save(fig, f"sauq_quickstart_{model_name.split('/')[-1]}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=24)
    args = p.parse_args()
    main(args.model, args.alpha, args.max_new_tokens)
