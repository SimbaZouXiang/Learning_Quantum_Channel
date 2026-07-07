"""Overlay learning-loss curves for weight-1 vs random Pauli training sets,
across N ∈ {8, 10, 12} × L_target ∈ {3..8} at MPO_layer=T=3. One subplot per
(N, L). X axis = epoch, Y axis = log-scale learning loss.

Saved to Figures/weight1_vs_random_learning_loss.png.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
RESULT_DIR = os.path.join(ROOT, "Learning_result")
FIG_DIR = os.path.join(ROOT, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

NS = [8, 10, 12]
LS = [3, 4, 5, 6, 7, 8]
T = 3


def load(N, L, tag):
    p = os.path.join(RESULT_DIR, f"W1RAND_N{N}_L{L}_T{T}_{tag}_learning_loss.npy")
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True).astype(float)


def load_test(N, L, tag):
    p = os.path.join(RESULT_DIR, f"W1RAND_N{N}_L{L}_T{T}_{tag}_testing_loss.npy")
    if not os.path.exists(p):
        return None
    return float(np.load(p, allow_pickle=True))


def main():
    fig, axes = plt.subplots(len(NS), len(LS),
                             figsize=(3.4 * len(LS), 2.8 * len(NS)),
                             sharex=False, sharey=False)
    if len(NS) == 1:
        axes = np.array([axes])
    for i, N in enumerate(NS):
        for j, L in enumerate(LS):
            ax = axes[i, j]
            curve_w1 = load(N, L, "w1")
            curve_rd = load(N, L, "rand")
            test_w1 = load_test(N, L, "w1")
            test_rd = load_test(N, L, "rand")
            if curve_w1 is not None:
                ax.plot(np.arange(1, len(curve_w1) + 1), curve_w1,
                        color="C0", label="weight-1")
            if curve_rd is not None:
                ax.plot(np.arange(1, len(curve_rd) + 1), curve_rd,
                        color="C1", label="random")
            ax.set_yscale("log")
            ax.set_title(f"N={N}, L={L}")
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlabel("epoch")
            ax.set_ylabel("training loss")
            tlbl = []
            if test_w1 is not None: tlbl.append(f"w1 test={test_w1:.3e}")
            if test_rd is not None: tlbl.append(f"rand test={test_rd:.3e}")
            if tlbl:
                ax.text(0.98, 0.02, "\n".join(tlbl),
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Weight-1 vs random Pauli training set "
        f"(MPO_layer=T={T}, 3N samples, same epochs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, "weight1_vs_random_learning_loss.png")
    for ext in ("png", "pdf"):
        out_e = out.replace(".png", f".{ext}")
        fig.savefig(out_e)
        print(f"Saved {out_e}")

    # Summary CSV
    csv = os.path.join(FIG_DIR, "weight1_vs_random_summary.csv")
    with open(csv, "w") as f:
        f.write("N,L,final_train_w1,final_train_rand,test_w1,test_rand,"
                "ratio_train_rand_over_w1,ratio_test_rand_over_w1\n")
        for N in NS:
            for L in LS:
                cw1 = load(N, L, "w1")
                crd = load(N, L, "rand")
                tw1 = load_test(N, L, "w1")
                trd = load_test(N, L, "rand")
                if cw1 is None or crd is None: continue
                rfw1 = cw1[-1]; rfrd = crd[-1]
                f.write(f"{N},{L},{rfw1:.6e},{rfrd:.6e},"
                        f"{tw1 if tw1 is not None else 'nan'},"
                        f"{trd if trd is not None else 'nan'},"
                        f"{rfrd/rfw1:.4f},"
                        f"{(trd/tw1) if (tw1 and trd is not None) else 'nan'}\n")
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
