"""Diagnose seed-to-seed variance of QMLM-with-bath training at small g.

For TL in {2, 3, 4, 5} and g in {0.05, 0.10, 0.15, 0.20} (200-epoch runs,
20-40 seeds each), plots:
  - histogram of final train-loss across seeds
  - all training trajectories overlaid (colour = good seed -> bad seed)
  - scatter of (epoch-1 loss after first gradient step) vs (final loss),
    to check if the lottery is decided in the first epoch.
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


def load_seeds(T_L, g_tag, bd):
    """Return list of (seed_offset, learning_loss_array, test_loss_scalar)."""
    out = []
    for llf in sorted(glob.glob(os.path.join(
            NPY_DIR,
            f"bath_sweep_N{N}_T{T_L}_L{T_L}_g{g_tag}_p010_bd{bd}_s*_e200_learning_loss.npy"))):
        seed = int(llf.rsplit("_s", 1)[1].split("_")[0])
        ll = np.load(llf)
        tlf = llf.replace("_learning_loss.npy", "_testing_loss.npy")
        tl = float(np.load(tlf)[0]) if os.path.exists(tlf) else float('nan')
        out.append((seed, ll, tl))
    return out


def main():
    # ---- Figure 1: histograms ----------------------------------------------
    fig, axes = plt.subplots(len(DEPTHS), len(G_VALS), figsize=(15, 10), sharex=False, sharey=False)
    for i, T_L in enumerate(DEPTHS):
        bd = 256 if T_L == 2 else 64
        for j, g in enumerate(G_VALS):
            g_tag = f"{int(round(g * 100)):03d}"
            ax = axes[i, j]
            data = load_seeds(T_L, g_tag, bd)
            finals = np.array([ll[-1] for _, ll, _ in data])
            if len(finals) == 0:
                ax.set_visible(False); continue
            ax.hist(finals, bins=20, color="steelblue", edgecolor="k", alpha=0.85)
            ax.axvline(finals.mean(), color="red", lw=2,
                       label=f"mean {finals.mean():.3f}")
            ax.axvline(np.median(finals), color="orange", lw=2, ls="--",
                       label=f"median {np.median(finals):.3f}")
            ax.set_title(f"T=L={T_L}, g={g} (n={len(finals)})", fontsize=11)
            ax.set_xlabel("final train loss")
            if j == 0:
                ax.set_ylabel("count")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("Distribution of final training loss across seeds (200 epochs)", fontsize=13)
    fig.tight_layout()
    out1 = os.path.join(FIG_DIR, f"seed_variance_N{N}_histograms.png")
    fig.savefig(out1, dpi=140); plt.close(fig)
    print(f"saved {out1}")

    # ---- Figure 2: training trajectories overlaid (sorted by final loss) ---
    fig, axes = plt.subplots(len(DEPTHS), len(G_VALS), figsize=(15, 10), sharex=True)
    for i, T_L in enumerate(DEPTHS):
        bd = 256 if T_L == 2 else 64
        for j, g in enumerate(G_VALS):
            g_tag = f"{int(round(g * 100)):03d}"
            ax = axes[i, j]
            data = load_seeds(T_L, g_tag, bd)
            if not data:
                ax.set_visible(False); continue
            # sort by final loss so the colourmap is monotone
            data_sorted = sorted(data, key=lambda x: x[1][-1])
            cmap = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(data_sorted)))
            for k, (seed, ll, _) in enumerate(data_sorted):
                ax.plot(range(len(ll)), ll, color=cmap[k], lw=0.6, alpha=0.7)
            ax.set_title(f"T=L={T_L}, g={g}", fontsize=11)
            ax.set_xlabel("epoch")
            if j == 0:
                ax.set_ylabel("train loss")
            ax.grid(alpha=0.3)
            ax.set_yscale("log")
    fig.suptitle("All training trajectories (blue = low final loss → red = high final loss)",
                 fontsize=13)
    fig.tight_layout()
    out2 = os.path.join(FIG_DIR, f"seed_variance_N{N}_trajectories.png")
    fig.savefig(out2, dpi=140); plt.close(fig)
    print(f"saved {out2}")

    # ---- Figure 3: is the outcome decided in early epochs? -----------------
    fig, axes = plt.subplots(len(DEPTHS), len(G_VALS), figsize=(15, 10))
    for i, T_L in enumerate(DEPTHS):
        bd = 256 if T_L == 2 else 64
        for j, g in enumerate(G_VALS):
            g_tag = f"{int(round(g * 100)):03d}"
            ax = axes[i, j]
            data = load_seeds(T_L, g_tag, bd)
            if not data:
                ax.set_visible(False); continue
            early = np.array([ll[5] if len(ll) > 5 else ll[-1] for _, ll, _ in data])
            late = np.array([ll[-1] for _, ll, _ in data])
            ax.scatter(early, late, s=20, alpha=0.7)
            mn = min(early.min(), late.min())
            mx = max(early.max(), late.max())
            ax.plot([mn, mx], [mn, mx], 'k--', lw=0.7, alpha=0.5)
            corr = np.corrcoef(early, late)[0, 1]
            ax.set_title(f"T=L={T_L}, g={g}  ρ={corr:.2f}", fontsize=11)
            ax.set_xlabel("loss at epoch 5")
            if j == 0:
                ax.set_ylabel("final loss (epoch 199)")
            ax.grid(alpha=0.3)
    fig.suptitle("Does the loss at epoch 5 predict the final loss?", fontsize=13)
    fig.tight_layout()
    out3 = os.path.join(FIG_DIR, f"seed_variance_N{N}_early_vs_final.png")
    fig.savefig(out3, dpi=140); plt.close(fig)
    print(f"saved {out3}")

    # ---- Quick summary statistics table ------------------------------------
    print()
    print("=== Per-(T_L, g) summary ===")
    print(f"{'T_L':>3} {'g':>5} {'n':>4} {'mean':>8} {'median':>8} {'std':>8} "
          f"{'min':>8} {'max':>8} {'iqr':>8}  modality?")
    for T_L in DEPTHS:
        bd = 256 if T_L == 2 else 64
        for g in G_VALS:
            g_tag = f"{int(round(g * 100)):03d}"
            data = load_seeds(T_L, g_tag, bd)
            if not data: continue
            finals = np.array([ll[-1] for _, ll, _ in data])
            q25, q75 = np.percentile(finals, [25, 75])
            # rough bimodality flag: ratio (max - min) / (q75 - q25) > 3 suggests
            # heavy tails or split distribution.
            tail_ratio = (finals.max() - finals.min()) / max(q75 - q25, 1e-9)
            modality = "bimodal/heavy" if tail_ratio > 4 else "unimodal-ish"
            print(f"{T_L:>3} {g:>5.2f} {len(finals):>4d} {finals.mean():>8.3f} "
                  f"{np.median(finals):>8.3f} {finals.std():>8.3f} "
                  f"{finals.min():>8.3f} {finals.max():>8.3f} "
                  f"{q75 - q25:>8.3f}  {modality}")


if __name__ == "__main__":
    main()
