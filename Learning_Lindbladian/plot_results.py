"""Plot learning + Trotterization results in Learning_result/.

Two kinds of files live side-by-side in Learning_result/:

    TDME_N{N}_T{T}_Modeltolearnlayer{ML}_time{t}_gamma{gg:03d}_{kind}.npy
        kind ∈ {learning_loss, testing_loss, testing_loss_list}
            * learning_loss      : 1D array, one loss per training epoch
            * testing_loss       : scalar, mean test loss of the trained model
            * testing_loss_list  : 1D array, per-sample test losses

    Trotterization_Testing_loss{_list,}_t{t}_gamma{g}_N{N}_T{T}_
    Modeltolearnlayer{ML}_mu{mu}_gamma{g}_t{t}.npy
            * Testing_loss       : scalar, Trotter-error baseline
            * Testing_loss_list  : 1D array, per-sample Trotter errors

Both families share the `gamma = gamma_name * 0.01` convention (so gamma_name=2
→ gamma=0.02).  For each target time t we plot:

    Fig. A — learning curves for every available gamma
    Fig. B — trained-model test loss vs gamma, overlaid with the
             Trotterization baseline (scalar + per-sample error bars / spread)

Figures land in Figures/.  Run as
    python plot_results.py
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "Learning_result")
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")

# ---------- filename parsers ------------------------------------------------

_TDME_RE = re.compile(
    r"^TDME_N(?P<N>\d+)_T(?P<T>\d+)_Modeltolearnlayer(?P<ML>\d+)"
    r"_time(?P<t>[\d.]+)_gamma(?P<gname>\d+)_(?P<kind>learning_loss|testing_loss_list|testing_loss)\.npy$"
)

_TROT_RE = re.compile(
    r"^Trotterization_Testing_loss(?P<list>_list)?_t(?P<t>[\d.]+)"
    r"_gamma(?P<gname>\d+)_N(?P<N>\d+)_T(?P<T>\d+)"
    r"_Modeltolearnlayer(?P<ML>\d+)_mu(?P<mu>[\d.]+)"
    r"_gamma\d+_t[\d.]+\.npy$"
)


def _scan_results(result_dir: str):
    """Group file paths by (N, T, ML, t, gamma_name)."""
    tdme = defaultdict(dict)   # (N,T,ML,t,gname) -> {kind: path}
    trot = defaultdict(dict)   # (N,T,ML,t,gname) -> {"scalar"|"list": path}
    for fn in sorted(os.listdir(result_dir)):
        full = os.path.join(result_dir, fn)
        if (m := _TDME_RE.match(fn)) is not None:
            key = (int(m["N"]), int(m["T"]), int(m["ML"]),
                   float(m["t"]), int(m["gname"]))
            tdme[key][m["kind"]] = full
        elif (m := _TROT_RE.match(fn)) is not None:
            key = (int(m["N"]), int(m["T"]), int(m["ML"]),
                   float(m["t"]), int(m["gname"]))
            trot[key]["list" if m["list"] else "scalar"] = full
    return tdme, trot


# ---------- plotting --------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "figure.figsize": (10, 6),
    })


def _mean_safe(arr):
    arr = np.asarray(arr)
    return float(arr.mean()) if arr.size else np.nan


def plot_learning_curves(tdme, outdir):
    """One figure per (N, T, ML, t): learning_loss curves for every gamma."""
    groups = defaultdict(dict)   # (N,T,ML,t) -> {gname: path}
    for key, kinds in tdme.items():
        if "learning_loss" not in kinds:
            continue
        *group_key, gname = key
        groups[tuple(group_key)][gname] = kinds["learning_loss"]

    for (N, T, ML, t), by_gamma in sorted(groups.items()):
        if not by_gamma:
            continue
        fig, ax = plt.subplots()
        gnames = sorted(by_gamma)
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(gnames)))
        for color, gname in zip(cmap, gnames):
            curve = np.load(by_gamma[gname])
            if curve.size == 0:
                continue
            ax.plot(curve, color=color,
                    label=rf"$\gamma = {gname * 0.01:.2f}$")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        ax.set_yscale("log")
        ax.set_title(f"Learning curves  N={N}, T={T}, "
                     f"model_to_learn_layer={ML}, t={t}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(ncol=2, loc="best")
        fig.tight_layout()
        fn = f"learning_curves_N{N}_T{T}_ML{ML}_t{t}.png"
        fig.savefig(os.path.join(outdir, fn), dpi=150)
        plt.close(fig)
        print(f"saved {fn}")


def plot_loss_vs_gamma(tdme, trot, outdir):
    """One figure per (N, T, ML, t): test loss of the trained model vs gamma,
    overlaid with the Trotterization baseline.  Uses per-sample lists when
    available to draw ±1σ bands."""
    groups = defaultdict(dict)   # (N,T,ML,t) -> {gname: {"tdme": (mean, list), "trot": (mean, list)}}
    for key, kinds in tdme.items():
        N, T, ML, t, gname = key
        entry = groups[(N, T, ML, t)].setdefault(gname, {})
        mean = np.load(kinds["testing_loss"]).item() if "testing_loss" in kinds else np.nan
        lst = np.load(kinds["testing_loss_list"]) if "testing_loss_list" in kinds else np.array([])
        entry["tdme"] = (mean, lst)
    for key, kinds in trot.items():
        N, T, ML, t, gname = key
        entry = groups[(N, T, ML, t)].setdefault(gname, {})
        mean = np.load(kinds["scalar"]).item() if "scalar" in kinds else np.nan
        lst = np.load(kinds["list"]) if "list" in kinds else np.array([])
        entry["trot"] = (mean, lst)

    for (N, T, ML, t), by_gamma in sorted(groups.items()):
        gnames = sorted(by_gamma)
        gammas = np.array([g * 0.01 for g in gnames])

        tdme_mean = np.array([by_gamma[g].get("tdme", (np.nan, []))[0] for g in gnames])
        tdme_std = np.array([
            float(np.asarray(by_gamma[g].get("tdme", (np.nan, np.array([])))[1]).std())
            if np.asarray(by_gamma[g].get("tdme", (np.nan, np.array([])))[1]).size else 0.0
            for g in gnames
        ])
        trot_mean = np.array([by_gamma[g].get("trot", (np.nan, []))[0] for g in gnames])
        trot_std = np.array([
            float(np.asarray(by_gamma[g].get("trot", (np.nan, np.array([])))[1]).std())
            if np.asarray(by_gamma[g].get("trot", (np.nan, np.array([])))[1]).size else 0.0
            for g in gnames
        ])

        if np.all(np.isnan(tdme_mean)) and np.all(np.isnan(trot_mean)):
            continue

        fig, ax = plt.subplots()
        if not np.all(np.isnan(tdme_mean)):
            ax.errorbar(gammas, tdme_mean, yerr=tdme_std, marker="o",
                        capsize=4, label="trained model (TDME)")
        if not np.all(np.isnan(trot_mean)):
            ax.errorbar(gammas, trot_mean, yerr=trot_std, marker="s",
                        capsize=4, label="Trotter baseline")
        ax.set_xlabel(r"dephasing strength $\gamma$")
        ax.set_ylabel("mean test loss")
        ax.set_yscale("log")
        ax.set_title(f"Test loss vs $\\gamma$  (N={N}, T={T}, "
                     f"model_to_learn_layer={ML}, t={t})")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best")
        fig.tight_layout()
        fn = f"loss_vs_gamma_N{N}_T{T}_ML{ML}_t{t}.png"
        fig.savefig(os.path.join(outdir, fn), dpi=150)
        plt.close(fig)
        print(f"saved {fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default=RESULT_DIR)
    ap.add_argument("--out-dir", default=FIG_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _apply_style()

    tdme, trot = _scan_results(args.result_dir)
    print(f"Found {len(tdme)} TDME learning entries, "
          f"{len(trot)} Trotterization entries")

    plot_learning_curves(tdme, args.out_dir)
    plot_loss_vs_gamma(tdme, trot, args.out_dir)
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
