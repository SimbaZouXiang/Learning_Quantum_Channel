"""Summarize the 20-cell asymmetry test on MPO targets.

Per (depol, L_target) cell, report 6 numbers in a CSV and produce a heatmap
of the cross-evaluation losses.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams['font.size'] = 11
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
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


def main():
    rows = []
    for p in DEPOL_LIST:
        for L in LTARGETS:
            # Train weight, eval set
            w1_on_w1 = _load(L, p, 1, "weight1")
            w1_on_w2 = _load(L, p, 1, "weight2")
            w1_on_rd = _load(L, p, 1, "random")
            w2_on_w1 = _load(L, p, 2, "weight1")
            w2_on_w2 = _load(L, p, 2, "weight2")
            w2_on_rd = _load(L, p, 2, "random")
            mean = lambda a: float(a.mean()) if a is not None else float('nan')
            median = lambda a: float(np.median(a)) if a is not None else float('nan')
            row = dict(
                p=p, L=L,
                w1_on_w1_mean=mean(w1_on_w1), w1_on_w1_median=median(w1_on_w1),
                w1_on_w2_mean=mean(w1_on_w2), w1_on_w2_median=median(w1_on_w2),
                w1_on_rd_mean=mean(w1_on_rd), w1_on_rd_median=median(w1_on_rd),
                w2_on_w1_mean=mean(w2_on_w1), w2_on_w1_median=median(w2_on_w1),
                w2_on_w2_mean=mean(w2_on_w2), w2_on_w2_median=median(w2_on_w2),
                w2_on_rd_mean=mean(w2_on_rd), w2_on_rd_median=median(w2_on_rd),
            )
            rows.append(row)
    # CSV
    csv = os.path.join(FD, "mpo_asymmetry_summary.csv")
    headers = ["p", "L",
               "w1_train_on_w1", "w1_train_on_w2", "w1_train_on_random",
               "w2_train_on_w1", "w2_train_on_w2", "w2_train_on_random",
               "ratio_w1train_w2over_w1", "ratio_w2train_w1over_w2"]
    with open(csv, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            w1_w1 = r["w1_on_w1_mean"]; w1_w2 = r["w1_on_w2_mean"]; w1_rd = r["w1_on_rd_mean"]
            w2_w1 = r["w2_on_w1_mean"]; w2_w2 = r["w2_on_w2_mean"]; w2_rd = r["w2_on_rd_mean"]
            ratio_w1train = w1_w2 / w1_w1 if w1_w1 > 0 else float('nan')
            ratio_w2train = w2_w1 / w2_w2 if w2_w2 > 0 else float('nan')
            f.write(f"{r['p']},{r['L']},{w1_w1:.4e},{w1_w2:.4e},{w1_rd:.4e},"
                    f"{w2_w1:.4e},{w2_w2:.4e},{w2_rd:.4e},"
                    f"{ratio_w1train:.3f},{ratio_w2train:.3f}\n")
    print(f"Saved {csv}")

    # Plot: 2x2 grid (p in rows, train-weight in cols). Each panel: L on x, eval-set on bars.
    fig, axes = plt.subplots(len(DEPOL_LIST), 2,
                              figsize=(11, 4.0 * len(DEPOL_LIST)),
                              squeeze=False)
    width = 0.25
    for pi, p in enumerate(DEPOL_LIST):
        for wi, w in enumerate((1, 2)):
            ax = axes[pi, wi]
            xs = np.arange(len(LTARGETS))
            for offset, eval_set, color in [(-width, "weight1", "C0"),
                                              (0.0, "weight2", "C1"),
                                              (width, "random", "C2")]:
                vals = []
                for L in LTARGETS:
                    a = _load(L, p, w, eval_set)
                    vals.append(a.mean() if a is not None else np.nan)
                ax.bar(xs + offset, vals, width=width, color=color,
                        label=f"eval: {eval_set}")
            ax.set_xticks(xs)
            ax.set_xticklabels(LTARGETS)
            ax.set_xlabel(r"$L_{\rm target}$")
            ax.set_ylabel("mean eval loss")
            ax.set_yscale("log")
            ax.set_title(f"train weight = {w},  $p = {p}$")
            ax.grid(True, alpha=0.25, which="both")
            ax.legend(fontsize=9)
    fig.suptitle(
        rf"MPO-target asymmetry test: $N={N}$, $T_{{model}}={T}$, 200-epoch trainings",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"mpo_asymmetry_bars.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
