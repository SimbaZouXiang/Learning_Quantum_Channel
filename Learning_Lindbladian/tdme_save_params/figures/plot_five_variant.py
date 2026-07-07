"""5-variant comparison for TDME Lindbladian learning.

For each time t in {0.5, 1.0, 2.0, 3.0}, plot testing loss vs gamma with one
line per training-input variant.
"""
import os
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
RD   = os.path.join(ROOT, "results")
FD   = HERE

N, T = 8, 3
L_TARGET = 10
T_LIST     = [0.5, 1.0, 2.0, 3.0]
GAMMA_LIST = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50]

VARIANTS = [
    ("w1 (24)",     "w1",        "C0", "o", "-"),
    ("w2 (252)",    "w2full",    "C2", "v", "-"),
    ("w1+w2 (276)", "combined",  "C3", "s", "-"),
    ("24 random",   "random24",  "C0", "^", "--"),
    ("276 random",  "random276", "C3", "D", "--"),
]


def _prefix(t, g, variant):
    g_int = int(round(g * 100))
    return os.path.join(RD, f"N{N}_T{T}_L{L_TARGET}_t{t}_g{g_int:03d}_{variant}")


def _load_testing(t, g, variant):
    """Return (mean, sem). NaN if missing."""
    fn = _prefix(t, g, variant) + "_testing_loss_list.npy"
    if not os.path.exists(fn):
        return float('nan'), float('nan')
    arr = np.load(fn, allow_pickle=True)
    if len(arr) == 0:
        return float('nan'), float('nan')
    mean = float(np.mean(arr))
    sem  = float(np.std(arr) / np.sqrt(len(arr)))
    return mean, sem


def plot_testing(yscale='linear'):
    fig, axes = plt.subplots(1, len(T_LIST),
                              figsize=(5.0 * len(T_LIST), 4.2),
                              sharex=True, squeeze=False)
    for ti, t in enumerate(T_LIST):
        ax = axes[0, ti]
        for label, tag, color, marker, ls in VARIANTS:
            means, sems = [], []
            for g in GAMMA_LIST:
                m, s = _load_testing(t, g, tag)
                means.append(m); sems.append(s)
            ax.errorbar(GAMMA_LIST, means, yerr=sems,
                        marker=marker, color=color, linestyle=ls,
                        label=label, markersize=7, capsize=2)
        ax.set_xlabel(r"$\gamma$", fontsize=18)
        if ti == 0:
            ax.set_ylabel(r"$\langle d_F(M_{\rm out}, M_{\rm target}) \rangle$",
                          fontsize=15)
        ax.set_title(rf"$t={t}$", fontsize=18)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(GAMMA_LIST)
        ax.tick_params(axis='x', labelrotation=45)
        if yscale == 'log':
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        if ti == 0:
            ax.legend(fontsize=11, loc='best')
    fig.suptitle(
        rf"Lindbladian learning: 5-variant testing loss "
        rf"($N={N}$, $T_{{\rm student}}={T}$, $L_{{\rm target}}={L_TARGET}$, $\mu=1$, "
        rf"dephasing noise)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    suffix = "_log" if yscale == 'log' else ""
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"five_variant_tdme{suffix}.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


def plot_learning_curves():
    """Final + middle epoch learning-loss snapshot per cell."""
    fig, axes = plt.subplots(1, len(T_LIST),
                              figsize=(5.0 * len(T_LIST), 4.2),
                              sharex=True, squeeze=False)
    for ti, t in enumerate(T_LIST):
        ax = axes[0, ti]
        for label, tag, color, marker, ls in VARIANTS:
            finals = []
            for g in GAMMA_LIST:
                fn = _prefix(t, g, tag) + "_learning_loss.npy"
                if not os.path.exists(fn):
                    finals.append(float('nan')); continue
                arr = np.load(fn, allow_pickle=True)
                finals.append(float(arr[-1]) if len(arr) > 0 else float('nan'))
            ax.plot(GAMMA_LIST, finals, marker=marker, color=color, linestyle=ls,
                    label=label, markersize=7)
        ax.set_xlabel(r"$\gamma$", fontsize=18)
        if ti == 0:
            ax.set_ylabel("Final training loss", fontsize=15)
        ax.set_title(rf"$t={t}$", fontsize=18)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(GAMMA_LIST)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylim(bottom=0)
        if ti == 0:
            ax.legend(fontsize=11, loc='best')
    fig.suptitle(
        rf"Lindbladian learning: final training loss "
        rf"($N={N}$, $T_{{\rm student}}={T}$, $L_{{\rm target}}={L_TARGET}$)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"five_variant_tdme_train.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


def print_summary():
    print("\nSummary: best variant per (t, gamma) by mean testing loss")
    print(f"{'t':>4} {'gamma':>6}  " + "  ".join(f"{v[0].split(' ')[0]:>10}" for v in VARIANTS) + "   best")
    counts = {v[1]: 0 for v in VARIANTS}
    for t in T_LIST:
        for g in GAMMA_LIST:
            vals = {tag: _load_testing(t, g, tag)[0] for _, tag, *_ in VARIANTS}
            best = min(vals, key=lambda k: vals[k] if not np.isnan(vals[k]) else float('inf'))
            counts[best] += 1
            line = f"{t:>4} {g:>6.2f}  " + "  ".join(
                f"{vals[tag]:>10.3e}" + ('*' if tag == best else ' ') for _, tag, *_ in VARIANTS
            )
            print(line + f"  {best}")
    print(f"\nBest variant counts (out of {len(T_LIST)*len(GAMMA_LIST)} cells):")
    for label, tag, *_ in VARIANTS:
        print(f"  {label:>14}:  {counts[tag]}")


def main():
    plot_testing(yscale='linear')
    plot_testing(yscale='log')
    plot_learning_curves()
    print_summary()


if __name__ == "__main__":
    main()
