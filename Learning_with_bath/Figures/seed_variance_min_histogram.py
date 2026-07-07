"""Plot histograms of the MINIMUM training loss reached over the trajectory,
for TL in {2,3,4,5} and g in {0.05, 0.10, 0.15, 0.20}, across all seeds at
BATH_EPOCHS=200 (240 total epochs with fine-tune).

Companion to seed_variance_N10_histograms.png which used the FINAL train loss.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(SCRIPT_DIR, "..", "npy_outputs")
FIG_DIR = SCRIPT_DIR

N = 10
DEPTHS = [2, 3, 4, 5]
G_VALS = [0.05, 0.10, 0.15, 0.20]


def load_mins(T_L, g_tag, bd):
    """Return array of per-seed min-over-trajectory training loss."""
    mins = []
    pat = os.path.join(
        NPY_DIR,
        f"bath_sweep_N{N}_T{T_L}_L{T_L}_g{g_tag}_p010_bd{bd}_s*_e200_learning_loss.npy")
    for llf in sorted(glob.glob(pat)):
        ll = np.load(llf)
        mins.append(float(np.min(ll)))
    return np.asarray(mins)


def main():
    fig, axes = plt.subplots(len(DEPTHS), len(G_VALS), figsize=(15, 10),
                             sharex=False, sharey=False)
    for i, T_L in enumerate(DEPTHS):
        bd = 256 if T_L == 2 else 64
        for j, g in enumerate(G_VALS):
            g_tag = f"{int(round(g * 100)):03d}"
            ax = axes[i, j]
            mins = load_mins(T_L, g_tag, bd)
            if len(mins) == 0:
                ax.set_visible(False); continue
            ax.hist(mins, bins=20, color="seagreen", edgecolor="k", alpha=0.85)
            ax.axvline(mins.mean(), color="red", lw=2,
                       label=f"mean {mins.mean():.3f}")
            ax.axvline(np.median(mins), color="orange", lw=2, ls="--",
                       label=f"median {np.median(mins):.3f}")
            ax.set_title(f"T=L={T_L}, g={g} (n={len(mins)})", fontsize=11)
            ax.set_xlabel("min training loss (over 240 epochs)")
            if j == 0:
                ax.set_ylabel("count")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("Distribution of MINIMUM training loss across seeds "
                 f"(N={N}, BATH_EPOCHS=200)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"seed_variance_N{N}_min_histograms.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")

    # Summary table
    print()
    print("=== Per-(T_L, g) summary of MIN training loss across seeds ===")
    print(f"{'T_L':>3} {'g':>5} {'n':>4} {'mean':>8} {'median':>8} {'std':>8} "
          f"{'min':>8} {'max':>8} {'q25':>8} {'q75':>8}")
    for T_L in DEPTHS:
        bd = 256 if T_L == 2 else 64
        for g in G_VALS:
            g_tag = f"{int(round(g * 100)):03d}"
            mins = load_mins(T_L, g_tag, bd)
            if len(mins) == 0:
                continue
            q25, q75 = np.percentile(mins, [25, 75])
            print(f"{T_L:>3} {g:>5.2f} {len(mins):>4d} {mins.mean():>8.3f} "
                  f"{np.median(mins):>8.3f} {mins.std():>8.3f} "
                  f"{mins.min():>8.3f} {mins.max():>8.3f} "
                  f"{q25:>8.3f} {q75:>8.3f}")


if __name__ == "__main__":
    main()
