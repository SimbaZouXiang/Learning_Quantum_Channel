"""Aggregate the 30-training seed-variance results into per-(cell, variant)
means ± std across 5 seeds, on each of three eval sets."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 200

ROOT = os.path.dirname(__file__)
RD = os.path.join(ROOT, "results")
FD = os.path.join(ROOT, "figures")
os.makedirs(FD, exist_ok=True)

N, T = 8, 3
CELLS = [(4, 0.05), (6, 0.05)]
N_SEEDS = 5
VARIANTS = ["w1", "w2", "w2full"]
EVAL_SETS = ["weight1", "weight2", "random"]


def _load(L, p, variant, seed, eval_set):
    fn = f"N{N}_T{T}_L{L}_p{p}_{variant}_s{seed}_eval_{eval_set}.npy"
    pth = os.path.join(RD, fn)
    if not os.path.exists(pth):
        return None
    return np.load(pth, allow_pickle=True).astype(float)


def _stats(L, p, variant, eval_set):
    """Return (mean across seeds of per-seed mean loss, std across seeds)."""
    per_seed_means = []
    for s in range(N_SEEDS):
        arr = _load(L, p, variant, 10000 + s, eval_set)
        if arr is None:
            continue
        per_seed_means.append(arr.mean())
    arr = np.array(per_seed_means, dtype=float)
    return float(arr.mean()), float(arr.std()), len(arr)


def main():
    # Print the headline table.
    print(f"{'L':>3}  {'p':>5}  {'variant':>7}  {'eval':>8}  "
          f"{'mean':>10}  {'std':>10}  {'n_seeds':>7}")
    for L, p in CELLS:
        for v in VARIANTS:
            for es in EVAL_SETS:
                m, s, n = _stats(L, p, v, es)
                print(f"{L:>3}  {p:>5}  {v:>7}  {es:>8}  "
                      f"{m:.4e}  {s:.4e}  {n:>7}")
        print()

    # CSV
    csv = os.path.join(FD, "seed_variance_summary.csv")
    with open(csv, "w") as f:
        f.write("L,p,variant,eval_set,mean,std,n_seeds\n")
        for L, p in CELLS:
            for v in VARIANTS:
                for es in EVAL_SETS:
                    m, s, n = _stats(L, p, v, es)
                    f.write(f"{L},{p},{v},{es},{m:.6e},{s:.6e},{n}\n")
    print(f"Saved {csv}")

    # Bar chart: per (L, eval_set), 3 bars (w1, w2, w2full) with std error bars.
    fig, axes = plt.subplots(len(CELLS), len(EVAL_SETS),
                              figsize=(5.0 * len(EVAL_SETS), 4.0 * len(CELLS)),
                              squeeze=False)
    width = 0.6
    variant_colors = {"w1": "C0", "w2": "C1", "w2full": "C3"}
    variant_labels = {
        "w1": "weight-1 train (24)",
        "w2": "weight-2 same-op (84)",
        "w2full": "weight-2 full (252)",
    }
    for ci, (L, p) in enumerate(CELLS):
        for ei, es in enumerate(EVAL_SETS):
            ax = axes[ci, ei]
            for i, v in enumerate(VARIANTS):
                m, s, n = _stats(L, p, v, es)
                ax.bar(i, m, width=width, yerr=s, capsize=8,
                        color=variant_colors[v], label=variant_labels[v])
            ax.set_xticks(range(len(VARIANTS)))
            ax.set_xticklabels(VARIANTS)
            ax.set_ylabel("mean eval loss across seeds")
            ax.set_title(rf"$L={L}$, $p={p}$, eval={es}")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(bottom=0)
            if ci == 0 and ei == 0:
                ax.legend(fontsize=9, loc="upper right")
    fig.suptitle(
        rf"Seed-averaged comparison "
        rf"($N={N}$, $T_{{\rm model}}={T}$, {N_SEEDS} seeds per variant, "
        rf"error bars = std across seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"seed_variance_bars.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
