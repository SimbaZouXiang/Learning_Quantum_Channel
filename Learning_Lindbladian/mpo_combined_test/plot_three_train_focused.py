"""Focused 3-line comparison: w1 only / full w2 only / combined (w1+w2-full)."""
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
RD_COMBO  = os.path.join(ROOT, "results")
RD_SAMEOP = os.path.join(os.path.dirname(ROOT), "mpo_asymmetry_test", "results")
RD_FULL   = os.path.join(os.path.dirname(ROOT), "mpo_asymmetry_full_test", "results")
FD = os.path.join(ROOT, "figures")
os.makedirs(FD, exist_ok=True)

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]


def _load(rd, tag, kind):
    pth = os.path.join(rd, f"{tag}_eval_{kind}.npy")
    return np.load(pth, allow_pickle=True).astype(float) if os.path.exists(pth) else None


def _mean(arr):
    return float(arr.mean()) if arr is not None else float('nan')


def main():
    eval_sets = [("weight1", "weight-1 eval"),
                  ("weight2", "weight-2 eval"),
                  ("random",  "random Pauli eval")]
    fig, axes = plt.subplots(len(DEPOL_LIST), len(eval_sets),
                              figsize=(5.0 * len(eval_sets), 4.0 * len(DEPOL_LIST)),
                              sharex=True, squeeze=False)
    for pi, p in enumerate(DEPOL_LIST):
        for ei, (kind, title) in enumerate(eval_sets):
            ax = axes[pi, ei]
            series = [
                ("w1 only (24 inputs)",      RD_SAMEOP, lambda L: f"N{N}_T{T}_L{L}_p{p}_w1",       "C0", "o", "-"),
                ("full w2 only (252 inputs)", RD_FULL,   lambda L: f"N{N}_T{T}_L{L}_p{p}_w2full",   "C3", "^", "-"),
                ("w1 + w2 combined (276)",   RD_COMBO,  lambda L: f"N{N}_T{T}_L{L}_p{p}_combined", "C2", "D", "--"),
            ]
            for label, rd, tag_fn, color, marker, ls in series:
                means = [_mean(_load(rd, tag_fn(L), kind)) for L in LTARGETS]
                ax.plot(LTARGETS, means, marker=marker, color=color, linestyle=ls, label=label)
            ax.set_xlabel(r"$L_{\rm target}$")
            ax.set_ylabel("mean eval loss")
            ax.set_title(rf"$p={p}$,  {title}")
            ax.grid(True, alpha=0.25)
            ax.set_xticks(LTARGETS)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=10)

    fig.suptitle(
        rf"w1 only  vs  full w2 only  vs  w1+w2 combined "
        rf"($N={N}$, $T_{{\rm model}}={T}$, MPO target)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"three_train_focused.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
