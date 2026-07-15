"""Extract empirical L_max(p, T) from the peak sweep and fit the p^(-1/2) law.

Run AFTER job 1915581/1915582 have populated Result_peak_sweep/:
    python analyze_peak_sweep.py
Produces:
    Result_peak_sweep/peak_summary.json
    <paper folder>/Empirical_Lmax_vs_p.pdf   (log-log L_max vs p + fit + predictions)
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Result_peak_sweep")
PAPER = os.path.join(HERE, "Learning_and_Compressing_a_Quantum_Circuit_with_Parametrized_Quantum_Circuits")
PAT = re.compile(r"PeakSweep_N(\d+)_T(\d+)_L(\d+)_p(\d+)_seed(\d+)_learning_loss\.npy")

vE = np.log(4)


def lstar_pred(p):
    q = 1 - 4 * p / 3
    return (-1 + np.sqrt(1 - 2 / np.log(q))) / 2


def main():
    data = {}
    for f in glob.glob(os.path.join(OUT, "PeakSweep_*_learning_loss.npy")):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        N, T, L, ptag, seed = (int(m.group(i)) for i in range(1, 6))
        p = ptag / 10000.0
        final = float(np.load(f)[-1])
        data.setdefault((T, p), {}).setdefault(seed, {})[L] = final

    summary = []
    for (T, p), by_seed in sorted(data.items()):
        peaks = []
        for seed, byL in by_seed.items():
            if len(byL) < 4:
                continue
            Ls = sorted(byL)
            peaks.append(Ls[int(np.argmax([byL[L] for L in Ls]))])
        if not peaks:
            continue
        summary.append(dict(T=T, p=p, n_seeds=len(peaks),
                            Lmax_mean=float(np.mean(peaks)),
                            Lmax_std=float(np.std(peaks)),
                            Lmax_pred_A=None))
        print(f"T={T} p={p}: L_max = {np.mean(peaks):.2f} +- {np.std(peaks):.2f} "
              f"({len(peaks)} seeds)  [L*_LOE pred = {lstar_pred(p):.2f}]")

    json.dump(summary, open(os.path.join(OUT, "peak_summary.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for T, c, mk in ((2, "#1f77b4", "o"), (3, "#d62728", "s")):
        rows = [r for r in summary if r["T"] == T and r["n_seeds"] >= 2]
        if not rows:
            continue
        ps = np.array([r["p"] for r in rows])
        Lm = np.array([r["Lmax_mean"] for r in rows])
        Le = np.array([r["Lmax_std"] for r in rows])
        ax.errorbar(ps, Lm, yerr=Le, fmt=mk, ms=4, capsize=2, color=c,
                    label=rf"measured $L_{{\max}}$, $T={T}$")
        # power-law fit L = A p^b in the unsaturated window (exclude the T+1
        # floor at large p and the finite-size ceiling ~7 at small p)
        sel = (Lm > T + 1.05) & (Lm < 6.9)
        if sel.sum() >= 3:
            w = 1.0 / np.maximum(Le[sel], 0.3) ** 2
            b, a = np.polyfit(np.log(ps[sel]), np.log(Lm[sel]), 1, w=w)
            ax.plot(ps, np.exp(a) * ps ** b, "-", lw=1, color=c, alpha=0.7,
                    label=rf"fit $\propto p^{{{b:.2f}}}$")
    pg = np.logspace(np.log10(0.002), np.log10(0.05), 100)
    ax.plot(pg, np.sqrt(3 / (8 * pg)), "k--", lw=1, label=r"$\sqrt{3/(8p)}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"per-layer depolarizing strength $p$")
    ax.set_ylabel(r"empirical loss-peak depth $L_{\max}$")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "Empirical_Lmax_vs_p.pdf"))
    print("wrote", os.path.join(PAPER, "Empirical_Lmax_vs_p.pdf"))


if __name__ == "__main__":
    main()
