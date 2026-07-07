"""Plot exact MPO-MPO Frobenius distances from direct_contraction_distances.csv.

Two panels (p=0.01, p=0.05), each plotting Frobenius distance ||M_student - M_target||_F
vs L_target for the 5 student variants. Companion plot of inter-student distances is
written to a second figure file.
"""
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

fsize = 12
tsize = 12
tdir = 'in'
major = 7.5
minor = 4.5

plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
plt.rcParams['figure.dpi'] = 400
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.25

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
CSV  = os.path.join(ROOT, "results", "direct_contraction_distances.csv")
FD   = HERE

DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]

VARIANTS = [
    ("w1 (24)",     "w1",        "C0", "o", "-"),
    ("w2 (252)",    "w2full",    "C2", "v", "-"),
    ("w1+w2 (276)", "combined",  "C3", "s", "-"),
    ("24 random",   "random24",  "C0", "^", "--"),
    ("276 random",  "random276", "C3", "D", "--"),
]


def _load():
    """Return dict[(L, p)] -> {(A, B): frob_sq}."""
    cells = defaultdict(dict)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            L = int(r['L']); p = float(r['p'])
            cells[(L, p)][(r['A'], r['B'])] = float(r['frob_sq'])
    return cells


def _get(pairs, a, b):
    if a == b: return 0.0
    return pairs.get((a, b), pairs.get((b, a)))


def plot_to_target(cells):
    fig, axes = plt.subplots(1, len(DEPOL_LIST), figsize=(12, 5), sharex=True, squeeze=False)
    for pi, p in enumerate(DEPOL_LIST):
        ax = axes[0, pi]
        for label, tag, color, marker, ls in VARIANTS:
            ys = []
            for L in LTARGETS:
                pairs = cells.get((L, p), {})
                fs = _get(pairs, tag, 'target')
                ys.append(np.sqrt(fs) if fs is not None else np.nan)
            ax.plot(LTARGETS, ys, marker=marker, color=color, linestyle=ls,
                    label=label, markersize=8)
        ax.set_xlabel(r"$L_{\rm target}$", fontsize=20)
        ax.set_ylabel(r"$\|M_{\rm student} - M_{\rm target}\|_F$", fontsize=18)
        ax.set_title(rf"$p_{{\rm depol}}={p}$", fontsize=20)
        ax.set_xticks(LTARGETS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=12)
        ax.set_ylim(bottom=0)
    fig.suptitle(r"Exact MPO-MPO Frobenius distance to target ($N=8$, $T_{\rm model}=3$)",
                 fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"direct_contraction_to_target.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


def plot_inter_student(cells):
    """For each variant pair (10 pairs), plot ||A - B||_F vs L, one panel per p."""
    STUDS = [v[1] for v in VARIANTS]
    LAB = {v[1]: v[0].split(' ')[0] for v in VARIANTS}
    pairs_list = [(STUDS[i], STUDS[j]) for i in range(len(STUDS)) for j in range(i+1, len(STUDS))]
    fig, axes = plt.subplots(1, len(DEPOL_LIST), figsize=(14, 6), sharex=True, squeeze=False)
    cmap = plt.get_cmap('tab10')
    for pi, p in enumerate(DEPOL_LIST):
        ax = axes[0, pi]
        for k, (a, b) in enumerate(pairs_list):
            ys = []
            for L in LTARGETS:
                fs = _get(cells.get((L, p), {}), a, b)
                ys.append(np.sqrt(fs) if fs is not None else np.nan)
            ax.plot(LTARGETS, ys, marker='o', color=cmap(k % 10),
                    label=f"{LAB[a]}-{LAB[b]}", markersize=6)
        ax.set_xlabel(r"$L_{\rm target}$", fontsize=20)
        ax.set_ylabel(r"$\|M_A - M_B\|_F$", fontsize=18)
        ax.set_title(rf"$p_{{\rm depol}}={p}$", fontsize=20)
        ax.set_xticks(LTARGETS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=10, ncol=2)
        ax.set_ylim(bottom=0)
    fig.suptitle(r"Inter-student Frobenius distance ($N=8$, $T_{\rm model}=3$)", fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"direct_contraction_inter_student.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


def main():
    cells = _load()
    plot_to_target(cells)
    plot_inter_student(cells)


if __name__ == "__main__":
    main()
