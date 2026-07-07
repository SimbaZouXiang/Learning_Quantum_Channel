"""Compare w1 train / w2-same-op train / w2-full train on three eval sets.

Reads w1 + w2-same-op data from ../mpo_asymmetry_test/results/ and
w2-full data from ./results/. Plots 2x3 grid: rows=depol strength,
cols=eval set, each panel has 3 lines (w1, w2-same-op, w2-full).
"""
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
RD_FULL    = os.path.join(ROOT, "results")
RD_SAMEOP  = os.path.join(os.path.dirname(ROOT), "mpo_asymmetry_test", "results")
FD = os.path.join(ROOT, "figures")
os.makedirs(FD, exist_ok=True)

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]


def _load(rd, tag, kind):
    pth = os.path.join(rd, f"{tag}_eval_{kind}.npy")
    if not os.path.exists(pth):
        return None
    return np.load(pth, allow_pickle=True).astype(float)


def _mean(arr):
    return float(arr.mean()) if arr is not None else float('nan')


def main():
    eval_sets = [("weight1", "weight-1 eval"),
                  ("weight2", "weight-2 (same-op) eval"),
                  ("random",  "random Pauli eval")]
    fig, axes = plt.subplots(len(DEPOL_LIST), len(eval_sets),
                              figsize=(5.0 * len(eval_sets), 4.0 * len(DEPOL_LIST)),
                              sharex=True, squeeze=False)

    for pi, p in enumerate(DEPOL_LIST):
        for ei, (kind, title) in enumerate(eval_sets):
            ax = axes[pi, ei]
            for label, rd, tag_fn, color, marker in [
                ("weight-1 train (24 in)",
                 RD_SAMEOP, lambda L: f"N{N}_T{T}_L{L}_p{p}_w1", "C0", "o"),
                ("weight-2 same-op train (84 in)",
                 RD_SAMEOP, lambda L: f"N{N}_T{T}_L{L}_p{p}_w2", "C1", "s"),
                ("weight-2 FULL train (252 in)",
                 RD_FULL, lambda L: f"N{N}_T{T}_L{L}_p{p}_w2full", "C3", "^"),
            ]:
                means = []
                for L in LTARGETS:
                    arr = _load(rd, tag_fn(L), kind)
                    means.append(_mean(arr))
                ax.plot(LTARGETS, means, marker=marker, color=color, label=label)
            ax.set_xlabel(r"$L_{\rm target}$")
            ax.set_ylabel("mean eval loss")
            ax.set_title(rf"$p={p}$,  {title}")
            ax.grid(True, alpha=0.25)
            ax.set_xticks(LTARGETS)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)

    fig.suptitle(
        rf"Three training-set comparison "
        rf"($N={N}$, $T_{{\rm model}}={T}$, MPO target, 200+50 epochs)",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"three_train_compare.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")

    # Headline CSV
    csv = os.path.join(FD, "three_train_compare_summary.csv")
    with open(csv, "w") as f:
        f.write("p,L,eval_set,w1_train_mean,w2_train_mean,w2full_train_mean\n")
        for p in DEPOL_LIST:
            for L in LTARGETS:
                for kind, _ in eval_sets:
                    a1 = _load(RD_SAMEOP, f"N{N}_T{T}_L{L}_p{p}_w1", kind)
                    a2 = _load(RD_SAMEOP, f"N{N}_T{T}_L{L}_p{p}_w2", kind)
                    af = _load(RD_FULL,   f"N{N}_T{T}_L{L}_p{p}_w2full", kind)
                    f.write(f"{p},{L},{kind},{_mean(a1):.4e},{_mean(a2):.4e},{_mean(af):.4e}\n")
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
