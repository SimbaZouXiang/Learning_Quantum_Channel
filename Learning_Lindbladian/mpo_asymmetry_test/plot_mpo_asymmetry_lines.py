"""Per-eval-set comparison plot: x = L_target, y = mean loss, two lines
(w1 train, w2 train) per subplot. One row per depolarizing strength."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 10
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
LTARGETS   = [3, 4, 5, 6, 7]


def _load(L, p, w, kind):
    fn = f"N{N}_T{T}_L{L}_p{p}_w{w}_eval_{kind}.npy"
    pth = os.path.join(RD, fn)
    if not os.path.exists(pth):
        return None
    return np.load(pth, allow_pickle=True).astype(float)


def _mean_sem(arr):
    if arr is None:
        return None, 0.0
    return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))


def main():
    eval_sets = [
        ("weight1", "weight-1 inputs (24)",  r"weight-1 eval ($w_2/w_1 \to w_1$)"),
        ("weight2", "weight-2 inputs (84)",  r"weight-2 eval ($w_2/w_1 \to w_2$)"),
        ("random",  "random Paulis (30)",   r"random eval ($w_2/w_1 \to $ random)"),
    ]
    fig, axes = plt.subplots(len(DEPOL_LIST), len(eval_sets),
                              figsize=(5.0 * len(eval_sets), 4.0 * len(DEPOL_LIST)),
                              sharex=True, squeeze=False)
    for pi, p in enumerate(DEPOL_LIST):
        for ei, (kind, _desc, title) in enumerate(eval_sets):
            ax = axes[pi, ei]
            for train_w, color, marker, label in [
                (1, "C0", "o", "train: weight-1"),
                (2, "C1", "s", "train: weight-2"),
            ]:
                means, sems = [], []
                for L in LTARGETS:
                    arr = _load(L, p, train_w, kind)
                    m, s = _mean_sem(arr)
                    means.append(m if m is not None else np.nan)
                    sems.append(s)
                means = np.array(means); sems = np.array(sems)
                ax.errorbar(LTARGETS, means, yerr=sems,
                             marker=marker, color=color, capsize=3, label=label)
            ax.set_xlabel(r"$L_{\rm target}$")
            ax.set_ylabel("mean eval loss")
            ax.set_title(rf"$p={p}$,  {title}")
            ax.grid(True, alpha=0.25)
            ax.set_xticks(LTARGETS)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)

    fig.suptitle(
        rf"MPO asymmetry: w1 vs w2 training, evaluated on three input sets"
        rf"  ($N={N}$, $T_{{\rm model}}={T}$)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"mpo_asymmetry_lines.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
