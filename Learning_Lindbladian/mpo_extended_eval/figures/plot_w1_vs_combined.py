"""Three contractions: w1-target, w1+w2-target, w1-(w1+w2)."""
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

fsize = 12
tsize = 12
plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 7.5
plt.rcParams['ytick.major.size'] = 7.5
plt.rcParams['figure.dpi'] = 400

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
CSV  = os.path.join(ROOT, "results", "direct_contraction_distances.csv")
FD   = HERE

DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]


def _load():
    cells = defaultdict(dict)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            L = int(r['L']); p = float(r['p'])
            cells[(L, p)][(r['A'], r['B'])] = float(r['frob_sq'])
    return cells


def _get(pairs, a, b):
    if a == b: return 0.0
    return pairs.get((a, b), pairs.get((b, a)))


def main():
    cells = _load()
    LINES = [
        (r"$\|M_{w1} - M_{\rm target}\|_F$",        "w1",       "target",   "C0", "o", "-"),
        (r"$\|M_{w1+w2} - M_{\rm target}\|_F$",     "combined", "target",   "C3", "s", "-"),
        (r"$\|M_{w1} - M_{w1+w2}\|_F$",             "w1",       "combined", "C2", "D", "--"),
    ]

    fig, axes = plt.subplots(1, len(DEPOL_LIST), figsize=(12, 5), sharex=True, squeeze=False)
    for pi, p in enumerate(DEPOL_LIST):
        ax = axes[0, pi]
        for label, a, b, color, marker, ls in LINES:
            ys = []
            for L in LTARGETS:
                fs = _get(cells.get((L, p), {}), a, b)
                ys.append(np.sqrt(fs) if fs is not None else np.nan)
            ax.plot(LTARGETS, ys, marker=marker, color=color, linestyle=ls,
                    label=label, markersize=8)
        ax.set_xlabel(r"$L_{\rm target}$", fontsize=20)
        ax.set_ylabel(r"Frobenius distance", fontsize=18)
        ax.set_title(rf"$p_{{\rm depol}}={p}$", fontsize=20)
        ax.set_xticks(LTARGETS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=12)
        ax.set_ylim(bottom=0)
    fig.suptitle(r"w1 vs.\ w1+w2 vs.\ target ($N=8$, $T_{\rm model}=3$)", fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"w1_vs_combined.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
