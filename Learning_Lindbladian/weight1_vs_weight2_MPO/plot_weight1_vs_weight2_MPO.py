"""Overlay 3 training curves per L_target: w1@200 vs w2@60 vs w2@200."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 7.5
plt.rcParams['ytick.major.size'] = 7.5
plt.rcParams['figure.dpi'] = 200

ROOT = os.path.dirname(__file__)
RD = os.path.join(ROOT, "results")
FD = os.path.join(ROOT, "figures")
os.makedirs(FD, exist_ok=True)

N, T = 8, 3
DEPOL = 0.01
LTARGETS = [3, 4, 5, 6, 7]

SERIES = [
    ("w1",      "C0", "-",  "weight-1, 200 epochs (24 inputs)"),
    ("w2_e60",  "C1", "-",  "weight-2, 60 epochs (84 inputs)"),
    ("w2_e200", "C3", "--", "weight-2, 200 epochs (84 inputs)"),
]


def _load(L, tag, kind):
    p = os.path.join(RD, f"N{N}_T{T}_L{L}_p{DEPOL}_{tag}_{kind}.npy")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def main():
    fig, axes = plt.subplots(1, len(LTARGETS),
                             figsize=(4.0 * len(LTARGETS), 4.0),
                             sharey=False)
    if len(LTARGETS) == 1:
        axes = [axes]
    summary = []
    for ax, L in zip(axes, LTARGETS):
        for tag, color, ls, label in SERIES:
            curve = _load(L, tag, "learning_loss")
            if curve is None:
                continue
            curve = np.asarray(curve, dtype=float)
            ax.plot(np.arange(1, len(curve) + 1), curve,
                    color=color, linestyle=ls, label=label)
            test_scalar = _load(L, tag, "testing_loss")
            test_list = _load(L, tag, "testing_loss_list")
            final_train = float(curve[-1])
            min_train = float(curve.min())
            test_val = float(test_scalar) if test_scalar is not None else float("nan")
            n_test = len(test_list) if test_list is not None else 0
            summary.append([L, tag, len(curve), final_train, min_train, test_val, n_test])
            print(f"L={L}  tag={tag:8s}  epochs={len(curve):3d}  "
                  f"final_train={final_train:.4e}  min_train={min_train:.4e}  "
                  f"testing={test_val:.4e} (n={n_test})")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        ax.set_title(rf"$L_\mathrm{{target}}={L}$")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=9)

    fig.suptitle(
        rf"Weight-1 vs weight-2 Pauli training set (MPO target, "
        rf"$N={N}$, $T={T}$, depolarizing $p={DEPOL}$)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"weight1_vs_weight2_MPO_curves.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")

    csv = os.path.join(FD, "weight1_vs_weight2_MPO_summary.csv")
    with open(csv, "w") as f:
        f.write("L_target,tag,total_epochs,final_train_loss,min_train_loss,"
                "testing_loss,n_test_samples\n")
        for row in summary:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
