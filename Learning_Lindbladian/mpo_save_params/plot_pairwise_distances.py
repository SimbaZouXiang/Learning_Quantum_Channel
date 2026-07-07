"""Plot the three pairwise MPO Frobenius distances vs L_target,
one panel per (p, eval_set). Linear y-scale so the order of magnitudes
of D1, D2, D3 are directly visible."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['figure.dpi'] = 200

ROOT = os.path.dirname(__file__)
csv_path = os.path.join(ROOT, "figures", "mpo_pairwise_distances.csv")

import csv
rows = []
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

DEPOL_LIST = [0.01, 0.05]
EVAL_SETS = ["weight1_full", "weight2_full", "random_500"]
EVAL_TITLES = ["weight-1 inputs (24)", "weight-2 inputs (84)", "random Paulis (500)"]
LTARGETS = [3, 4, 5, 6, 7]

fig, axes = plt.subplots(len(DEPOL_LIST), len(EVAL_SETS),
                         figsize=(5.0 * len(EVAL_SETS), 4.0 * len(DEPOL_LIST)),
                         sharex=True, squeeze=False)

for pi, p in enumerate(DEPOL_LIST):
    for ei, (es, title) in enumerate(zip(EVAL_SETS, EVAL_TITLES)):
        ax = axes[pi, ei]
        D1, D2, D3 = [], [], []
        for L in LTARGETS:
            cell = next((r for r in rows
                         if int(r["L"]) == L
                         and float(r["p"]) == p
                         and r["eval_set"] == es), None)
            if cell is None:
                D1.append(np.nan); D2.append(np.nan); D3.append(np.nan)
            else:
                D1.append(float(cell["D_w1_target"]))
                D2.append(float(cell["D_combined_target"]))
                D3.append(float(cell["D_w1_combined"]))
        ax.plot(LTARGETS, D1, "o-", color="C0",
                label=r"$\|M_{w1}-M_{\rm tgt}\|^2$  (w1 student vs target)")
        ax.plot(LTARGETS, D2, "s-", color="C3",
                label=r"$\|M_{cmb}-M_{\rm tgt}\|^2$  (combined student vs target)")
        ax.plot(LTARGETS, D3, "D--", color="C2",
                label=r"$\|M_{w1}-M_{cmb}\|^2$  (between the two students)")
        ax.set_xlabel(r"$L_{\rm target}$")
        ax.set_ylabel("avg squared Frob. distance")
        ax.set_title(rf"$p={p}$,  {title}")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(LTARGETS)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)

fig.suptitle(
    r"Pairwise MPO distances: w1-trained student, combined-trained student, and target"
    "\n"
    r"($N=8$, $T_{\rm model}=3$, MPO target with depolarizing noise $p$)",
    fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "pdf"):
    out = os.path.join(ROOT, "figures", f"pairwise_distances.{ext}")
    fig.savefig(out)
    print(f"Saved {out}")
