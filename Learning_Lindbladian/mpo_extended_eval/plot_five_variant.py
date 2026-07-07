"""5-variant comparison plot using extended-eval data."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['figure.dpi'] = 200

ROOT = os.path.dirname(__file__)
RD = os.path.join(ROOT, "results")
FD = os.path.join(ROOT, "figures")
os.makedirs(FD, exist_ok=True)

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS = [3, 4, 5, 6, 7]

EVAL_SETS = [
    ("weight1full", "weight-1 inputs (24)"),
    ("weight2full", "weight-2 inputs (252)"),
    ("random500",   "random Paulis (500)"),
]

VARIANTS = [
    ("w1 (24, structured)",        "w1",        "C0", "o", "-"),
    ("w2-full (252, structured)",  "w2full",    "C2", "v", "-"),
    ("combined (276, structured)", "combined",  "C3", "s", "-"),
    ("random 24",                   "random24",  "C0", "^", "--"),
    ("random 276",                  "random276", "C3", "D", "--"),
]


def _mean(L, p, variant, es):
    fn = os.path.join(RD, f"N{N}_T{T}_L{L}_p{p}_{variant}_eval_{es}.npy")
    return float(np.load(fn, allow_pickle=True).mean()) if os.path.exists(fn) else float('nan')


def main():
    fig, axes = plt.subplots(len(DEPOL_LIST), len(EVAL_SETS),
                              figsize=(5.0 * len(EVAL_SETS), 4.0 * len(DEPOL_LIST)),
                              sharex=True, squeeze=False)
    for pi, p in enumerate(DEPOL_LIST):
        for ei, (es, title) in enumerate(EVAL_SETS):
            ax = axes[pi, ei]
            for label, tag, color, marker, ls in VARIANTS:
                means = [_mean(L, p, tag, es) for L in LTARGETS]
                ax.plot(LTARGETS, means, marker=marker, color=color,
                        linestyle=ls, label=label)
            ax.set_xlabel(r"$L_{\rm target}$")
            ax.set_ylabel("mean eval loss")
            ax.set_title(rf"$p={p}$,  {title}")
            ax.grid(True, alpha=0.25)
            ax.set_xticks(LTARGETS)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)

    fig.suptitle(
        rf"All 5 training variants compared on full-precision eval sets "
        rf"($N={N}$, $T_{{\rm model}}={T}$, MPO target)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"five_variant_extended.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
