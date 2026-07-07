"""Summary plots for the Lindbladian-learning comparison.

Reads the (already-generated) testing-loss .npy files and produces:
  Figures/heatmap_pqc_vs_trotter2.png  — log-ratio (2nd-order/PQC), t × gamma
  Figures/heatmap_pqc_vs_trotter1.png  — log-ratio (1st-order/PQC), t × gamma
  Figures/loss_vs_t_at_fixed_gamma.png — testing loss vs t for selected γ

Reading guide:
  Heatmap value = log10(Trotter_loss / PQC_loss).
    > 0 (red) → PQC wins (Trotter loss higher)
    < 0 (blue) → Trotter wins
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(ROOT, "Learning_result")
FD = os.path.join(ROOT, "Figures")
os.makedirs(FD, exist_ok=True)

TIMES = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
GAMMA_NAMES = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]


def load_pqc(t, gname):
    p = os.path.join(RD, f"TDME_N30_T3_Modeltolearnlayer30_time{t}_gamma{gname:03d}_testing_loss_list.npy")
    if not os.path.exists(p):
        return None
    return np.asarray(np.load(p, allow_pickle=True), dtype=float)


def load_trott_2(t, gname):
    pat = (f"Trotterization_Testing_loss_list_t{t}_gamma{gname}_N30_T3"
           f"_Modeltolearnlayer30_mu1.0_gamma*_t{t}.npy")
    matches = glob.glob(os.path.join(RD, pat))
    matches.sort(key=lambda p: 0 if f"_gamma{gname}_t" in p else 1)
    if not matches:
        return None
    return np.asarray(np.load(matches[0], allow_pickle=True), dtype=float)


def load_trott_1(t, gname):
    p = os.path.join(RD, f"FirstOrder_Trotterization_Testing_loss_list_t{t}_gamma{gname}"
                          f"_N30_T3_Modeltolearnlayer30_mu1.0_gamma{gname}_t{t}.npy")
    if not os.path.exists(p):
        return None
    return np.asarray(np.load(p, allow_pickle=True), dtype=float)


def _build_grid():
    """Return three (T,G)-shape arrays of mean testing loss for PQC, 1st, 2nd."""
    pqc  = np.full((len(TIMES), len(GAMMA_NAMES)), np.nan)
    tr1  = np.full((len(TIMES), len(GAMMA_NAMES)), np.nan)
    tr2  = np.full((len(TIMES), len(GAMMA_NAMES)), np.nan)
    for i, t in enumerate(TIMES):
        for j, g in enumerate(GAMMA_NAMES):
            ap = load_pqc(t, g);  pqc[i, j] = ap.mean() if ap is not None else np.nan
            a1 = load_trott_1(t, g); tr1[i, j] = a1.mean() if a1 is not None else np.nan
            a2 = load_trott_2(t, g); tr2[i, j] = a2.mean() if a2 is not None else np.nan
    return pqc, tr1, tr2


def _heatmap(ax, M, title, *, cbar_label):
    masked = np.ma.masked_invalid(M)
    vmax = np.nanmax(np.abs(masked))
    im = ax.imshow(masked, aspect="auto", origin="lower",
                   cmap="RdBu_r",
                   norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_xticks(range(len(GAMMA_NAMES)))
    ax.set_xticklabels([f"{g*0.01:.2f}" for g in GAMMA_NAMES], rotation=45)
    ax.set_yticks(range(len(TIMES)))
    ax.set_yticklabels([f"{t}" for t in TIMES])
    ax.set_xlabel(r"$\gamma$ (depolarizing strength)")
    ax.set_ylabel(r"$t$ (target evolution time)")
    ax.set_title(title)
    # Annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7, color="black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)


def main():
    pqc, tr1, tr2 = _build_grid()

    # log10 ratios
    log_r2_over_pqc = np.log10(tr2 / pqc)
    log_r1_over_pqc = np.log10(tr1 / pqc)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _heatmap(axes[0], log_r1_over_pqc,
             "1st-order Trotter vs PQC\nlog10(Trott1 / PQC); + → PQC better",
             cbar_label="log10(Trott1 / PQC)")
    _heatmap(axes[1], log_r2_over_pqc,
             "2nd-order Trotter vs PQC\nlog10(Trott2 / PQC); + → PQC better",
             cbar_label="log10(Trott2 / PQC)")
    fig.suptitle("Learning a 30-layer 2nd-order TDME Lindbladian with a 3-layer ansatz "
                 "(N=30, depolarizing noise)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = os.path.join(FD, "heatmap_pqc_vs_trotter.png")
    for ext in ("png", "pdf"):
        out_e = out1.replace(".png", f".{ext}")
        fig.savefig(out_e)
        print(f"Saved {out_e}")

    # ── Loss vs t at fixed gamma ───────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fixed_gammas = [0, 2, 10, 50]  # gamma_name
    for ax, gname in zip(axes2.flatten(), fixed_gammas):
        j = GAMMA_NAMES.index(gname)
        ax.plot(TIMES, pqc[:, j], "o-", color="C0", label="PQC")
        ax.plot(TIMES, tr1[:, j], "^-", color="C2", label="1st-order Trotter")
        ax.plot(TIMES, tr2[:, j], "s-", color="C3", label="2nd-order Trotter")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("avg testing loss")
        ax.set_title(rf"$\gamma$ = {gname * 0.01:.2f}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig2.suptitle("Testing loss vs t (3-layer ansatz, 30-layer 2nd-order TDME target)",
                  fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(FD, "loss_vs_t_at_fixed_gamma.png")
    for ext in ("png", "pdf"):
        out_e = out2.replace(".png", f".{ext}")
        fig2.savefig(out_e)
        print(f"Saved {out_e}")

    # ── Loss vs gamma at fixed t ───────────────────────────────────────
    fig3, axes3 = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
    fixed_ts = [0.2, 1.0, 3.0, 6.0]
    for ax, t in zip(axes3.flatten(), fixed_ts):
        i = TIMES.index(t)
        gx = np.array([g * 0.01 for g in GAMMA_NAMES])
        ax.plot(gx, pqc[i, :], "o-", color="C0", label="PQC")
        ax.plot(gx, tr1[i, :], "^-", color="C2", label="1st-order Trotter")
        ax.plot(gx, tr2[i, :], "s-", color="C3", label="2nd-order Trotter")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("avg testing loss")
        ax.set_title(f"t = {t}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig3.suptitle("Testing loss vs gamma (3-layer ansatz, 30-layer 2nd-order TDME target)",
                  fontsize=12)
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    out3 = os.path.join(FD, "loss_vs_gamma_at_fixed_t.png")
    for ext in ("png", "pdf"):
        out_e = out3.replace(".png", f".{ext}")
        fig3.savefig(out_e)
        print(f"Saved {out_e}")


if __name__ == "__main__":
    main()
