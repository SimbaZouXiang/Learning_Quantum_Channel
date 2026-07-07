"""5-variant comparison plot using extended-eval data."""
import os
import numpy as np
import matplotlib.pyplot as plt

'''plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['figure.dpi'] = 200'''

#%matplotlib inline
#for 3D non-interactive plots

#set up the size and DPI for the graph
fsize = 12
tsize = 12

tdir = 'in'
major = 7.5
minor = 4.5

style = 'default'

plt.style.use(style)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams['axes.grid']=False
plt.rcParams['grid.alpha'] = 0.25
#mpl.rcParams.update({"axes.grid" : True, "grid.alpha": 0.25})
plt.rcParams['figure.dpi'] = 400
plt.rcParams['text.usetex'] = True


# Paths are relative to mpo_extended_eval/figures/ (where this script lives).
HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)            # → mpo_extended_eval/
RD = os.path.join(ROOT, "results")      # eval-loss .npy files
FD = HERE                                # write figures alongside this script
os.makedirs(FD, exist_ok=True)

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS = [3, 4, 5, 6, 7]

EVAL_SETS = [
    ("weight1full", "weight-1 Paulis (24)"),
    ("weight2full", "weight-2 Paulis (252)"),
    ("random500",   "random Paulis (500)"),
]

VARIANTS = [
    ("w1 (24)",        "w1",        "C0", "o", "-"),
    ("w2 (252)",  "w2full",    "C2", "v", "-"),
    ("w1+w2 (276)", "combined",  "C3", "s", "-"),
    ("24 random",                   "random24",  "C0", "^", "--"),
    ("276 random",                  "random276", "C3", "D", "--"),
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
            ax.set_xlabel(r"$L_{\rm target}$", fontsize=20)
            ax.set_ylabel(r'$d_F(M_{out}, M_{target})$', fontsize=20)
            ax.set_title(rf"$p_{{\rm depol}}={p}$,  {title}", fontsize=20)
            ax.grid(True, alpha=0.25)
            ax.set_xticks(np.array([3, 4, 5, 6, 7]))
            ax.set_xticklabels(np.array([3, 4, 5, 6, 7]))
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=12)

    fig.suptitle(
        rf"Testing Loss for 5 Variants of Training Data Sets " +
        rf"($N={N}$, $T_{{\rm model}}={T}$)", fontsize=28)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"five_variant_extended.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
