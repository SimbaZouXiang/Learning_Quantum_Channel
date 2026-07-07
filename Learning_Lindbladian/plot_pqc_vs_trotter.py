"""Compare 3-layer PQC vs 3-layer 1st-order Trotter vs 3-layer 2nd-order Trotter,
all approximating the same 30-layer second-order TDME (Lindbladian) target.

The 3-layer 1st-order Trotter has the same per-layer "shape" as a typical PQC
(one noise sublayer + odd brickwall + even brickwall; no Strang symmetrization),
so it is the apples-to-apples Trotter baseline at fixed circuit depth.

For each target time t in {0.2, 0.4, 0.6, 0.8, 1.0, 2.0} we plot mean testing
loss vs gamma over 300 random Pauli-string inputs (PQC: 30-200, depending on
shard). Saved to Figures/pqc_vs_trotter.png and CSV in Figures/.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ── publication style ────────────────────────────────────────────────
fsize = 12
tsize = 12
tdir = 'in'
major = 7.5
minor = 4.5

plt.style.use('default')
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
plt.rcParams['figure.figsize'] = (24, 9)
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['figure.dpi'] = 400
plt.rcParams['text.usetex'] = True

RESULT_DIR = os.path.join(os.path.dirname(__file__), "Learning_result")
FIG_DIR = os.path.join(os.path.dirname(__file__), "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

TIMES = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
GAMMAS = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]


def load_pqc(t, gname):
    fn = (f"TDME_N30_T3_Modeltolearnlayer30_time{t}_gamma{gname:03d}"
          f"_testing_loss_list.npy")
    p = os.path.join(RESULT_DIR, fn)
    if not os.path.exists(p):
        return None
    return np.asarray(np.load(p, allow_pickle=True), dtype=float)


def load_pqc_min_train(t, gname):
    """Minimum training loss across all epochs for the PQC at this (t, γ)."""
    fn = (f"TDME_N30_T3_Modeltolearnlayer30_time{t}_gamma{gname:03d}"
          f"_learning_loss.npy")
    p = os.path.join(RESULT_DIR, fn)
    if not os.path.exists(p):
        return None
    arr = np.asarray(np.load(p, allow_pickle=True), dtype=float)
    return float(arr.min())


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
    n_cols = 5
    n_rows = (len(TIMES) + n_cols - 1) // n_cols
    # figsize comes from rcParams (16, 9) which already matches a 2x5 grid.
    fig, axes = plt.subplots(n_rows, n_cols, sharey=False)
    axes = axes.flatten()

    for ax, t in zip(axes, TIMES):
        gammas_x = []
        rows = {"pqc": [], "t1": [], "t2": []}
        pqc_min_train = []

        for g in GAMMAS:
            ap = load_pqc(t, g)
            a1 = load_trott_1st(t, g)
            a2 = load_trott_2nd(t, g)
            if ap is None and a1 is None and a2 is None:
                continue
            gammas_x.append(g * 0.01)
            rows["pqc"].append(_summary(ap))
            rows["t1"].append(_summary(a1))
            rows["t2"].append(_summary(a2))
            pqc_min_train.append(load_pqc_min_train(t, g))

        gammas_x = np.array(gammas_x)

        for key, color, marker, label in [
            ("pqc", "C0", "o", r"PQC, T=3, testing"),
            ("t1", "C2", "^", r"1st-order Trotter, T=3"),
            ("t2", "C3", "s", r"2nd-order Trotter, T=3"),
        ]:
            means = np.array([r[0] if r[0] is not None else np.nan for r in rows[key]])
            sems  = np.array([r[1] if r[1] is not None else 0.0 for r in rows[key]])
            ns    = [r[2] for r in rows[key] if r[2] > 0]
            if not np.any(np.isfinite(means)):
                continue
            ax.errorbar(gammas_x, means, yerr=sems,
                        marker=marker, color=color, capsize=3,
                        label=label)

        # PQC minimum training loss — same color (C0) as PQC test, dashed
        # so it visually pairs with its testing-loss counterpart.
        mins = np.array([m if m is not None else np.nan for m in pqc_min_train])
        if np.any(np.isfinite(mins)):
            ax.plot(gammas_x, mins, marker="x", linestyle="--", color="C0",
                    alpha=0.85, label=r"PQC, T=3, training")

        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"avg testing loss")
        ax.set_title(rf"$t = {t}$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=tsize - 2)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        r"3-layer PQC vs 1st-order vs 2nd-order Trotter "
        r"($N=30$, target = 30-layer 2nd-order TDME)", fontsize=fsize + 18)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"pqc_vs_trotter.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")

    csv_path = os.path.join(FIG_DIR, "pqc_vs_trotter_means.csv")
    with open(csv_path, "w") as f:
        f.write(
            "t,gamma,"
            "pqc_mean,pqc_sem,n_pqc,pqc_min_train,"
            "trott1_mean,trott1_sem,n_trott1,"
            "trott2_mean,trott2_sem,n_trott2,"
            "ratio_trott1_over_pqc,ratio_trott2_over_pqc\n"
        )
        for t in TIMES:
            for g in GAMMAS:
                ap = load_pqc(t, g)
                a1 = load_trott_1st(t, g)
                a2 = load_trott_2nd(t, g)
                if ap is None and a1 is None and a2 is None:
                    continue
                pm, ps, npc = _summary(ap)
                t1m, t1s, n1 = _summary(a1)
                t2m, t2s, n2 = _summary(a2)
                pmin = load_pqc_min_train(t, g)
                f.write(
                    f"{t},{g*0.01},"
                    f"{pm if pm is not None else 'nan'},"
                    f"{ps if ps is not None else 'nan'},{npc},"
                    f"{pmin if pmin is not None else 'nan'},"
                    f"{t1m if t1m is not None else 'nan'},"
                    f"{t1s if t1s is not None else 'nan'},{n1},"
                    f"{t2m if t2m is not None else 'nan'},"
                    f"{t2s if t2s is not None else 'nan'},{n2},"
                    f"{(t1m/pm) if (pm and t1m) else 'nan'},"
                    f"{(t2m/pm) if (pm and t2m) else 'nan'}\n"
                )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
