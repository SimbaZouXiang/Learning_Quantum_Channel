"""Compare 3-layer 1st-order vs 3-layer 2nd-order Trotter, both approximating
a 30-layer 2nd-order TDME (Lindbladian) target. One panel per target time.

Saved to Figures/trotter_first_vs_second.png.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), "Learning_result")
FIG_DIR = os.path.join(os.path.dirname(__file__), "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

TIMES = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
GAMMAS = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]


def load_trott_2nd(t, gname):
    pattern = (f"Trotterization_Testing_loss_list_t{t}_gamma{gname}"
               f"_N30_T3_Modeltolearnlayer30_mu1.0_gamma*_t{t}.npy")
    matches = glob.glob(os.path.join(RESULT_DIR, pattern))
    matches.sort(key=lambda p: 0 if f"_gamma{gname}_t" in p else 1)
    if not matches:
        return None
    return np.asarray(np.load(matches[0], allow_pickle=True), dtype=float)


def load_trott_1st(t, gname):
    fn = (f"FirstOrder_Trotterization_Testing_loss_list_t{t}_gamma{gname}"
          f"_N30_T3_Modeltolearnlayer30_mu1.0_gamma{gname}_t{t}.npy")
    p = os.path.join(RESULT_DIR, fn)
    if not os.path.exists(p):
        return None
    return np.asarray(np.load(p, allow_pickle=True), dtype=float)


def _summary(arr):
    if arr is None:
        return None, None, 0
    return arr.mean(), arr.std() / np.sqrt(len(arr)), len(arr)


def main():
    n_panels = len(TIMES)
    n_cols = 5
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                              sharey=False)
    axes = axes.flatten()

    for ax, t in zip(axes, TIMES):
        gammas_x = []
        rows_1, rows_2 = [], []
        for g in GAMMAS:
            a1 = load_trott_1st(t, g)
            a2 = load_trott_2nd(t, g)
            if a1 is None and a2 is None:
                continue
            gammas_x.append(g * 0.01)
            rows_1.append(_summary(a1))
            rows_2.append(_summary(a2))

        gammas_x = np.array(gammas_x)
        for rows, color, marker, label in [
            (rows_1, "C2", "^", "1st-order Trotter"),
            (rows_2, "C3", "s", "2nd-order Trotter"),
        ]:
            means = np.array([r[0] if r[0] is not None else np.nan for r in rows])
            sems  = np.array([r[1] if r[1] is not None else 0.0 for r in rows])
            ns    = [r[2] for r in rows if r[2] > 0]
            if not np.any(np.isfinite(means)):
                continue
            ax.errorbar(gammas_x, means, yerr=sems,
                        marker=marker, color=color, capsize=3,
                        label=f"{label} (n≈{int(np.mean(ns)) if ns else 0})")

        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("avg testing loss")
        ax.set_title(f"t = {t}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    # Hide any unused panels
    for k in range(len(TIMES), len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        "3-layer 1st-order vs 2nd-order Trotter "
        "(N=30, target = 30-layer 2nd-order TDME)", fontsize=30)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"trotter_first_vs_second.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")

    csv_path = os.path.join(FIG_DIR, "trotter_first_vs_second.csv")
    with open(csv_path, "w") as f:
        f.write(
            "t,gamma,"
            "trott1_mean,trott1_sem,n_trott1,"
            "trott2_mean,trott2_sem,n_trott2,"
            "ratio_trott1_over_trott2\n"
        )
        for t in TIMES:
            for g in GAMMAS:
                a1 = load_trott_1st(t, g)
                a2 = load_trott_2nd(t, g)
                if a1 is None and a2 is None:
                    continue
                t1m, t1s, n1 = _summary(a1)
                t2m, t2s, n2 = _summary(a2)
                ratio = (t1m / t2m) if (t1m is not None and t2m) else None
                f.write(
                    f"{t},{g*0.01},"
                    f"{t1m if t1m is not None else 'nan'},"
                    f"{t1s if t1s is not None else 'nan'},{n1},"
                    f"{t2m if t2m is not None else 'nan'},"
                    f"{t2s if t2s is not None else 'nan'},{n2},"
                    f"{ratio if ratio is not None else 'nan'}\n"
                )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
