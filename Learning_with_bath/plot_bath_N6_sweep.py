"""Plot training curves and final losses for the g-sweep at N=6, T=3, L=6."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(SCRIPT_DIR, "npy_outputs")
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

COUPLINGS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N, T, L, p = 6, 3, 6, 10

loss_curves = {}
final_train = {}
final_test = {}
for g in COUPLINGS:
    g_tag = f"{int(round(g * 100)):03d}"
    prefix = f"bath_sweep_N{N}_T{T}_L{L}_g{g_tag}_p{p:03d}"
    ll = np.load(os.path.join(NPY_DIR, f"{prefix}_learning_loss.npy"))
    tl = np.load(os.path.join(NPY_DIR, f"{prefix}_testing_loss.npy"))
    loss_curves[g] = ll
    final_train[g] = ll[-1]
    final_test[g] = float(tl[0])

# 1) overlay of loss curves
fig, ax = plt.subplots(figsize=(9, 5.5))
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(COUPLINGS)))
for g, color in zip(COUPLINGS, cmap):
    ll = loss_curves[g]
    ax.plot(range(len(ll)), ll, label=f"g = {g:.2f}", color=color, linewidth=1.5)
ax.set_xlabel("epoch")
ax.set_ylabel("train loss (tensor_network_distance)")
ax.set_yscale("log")
ax.set_title(f"QMLM-with-bath: loss curves vs coupling strength (N={N}, T={T}, L={L})")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"bath_sweep_N{N}_T{T}_L{L}_loss_curves.png"), dpi=150)
print(f"saved Figures/bath_sweep_N{N}_T{T}_L{L}_loss_curves.png")

# 2) final train + test loss vs coupling
gs = np.array(COUPLINGS)
tr = np.array([final_train[g] for g in COUPLINGS])
te = np.array([final_test[g] for g in COUPLINGS])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(gs, tr, "o-", label="final train loss")
ax.plot(gs, te, "s-", label="avg test loss")
ax.set_xlabel("coupling strength  g")
ax.set_ylabel("loss")
ax.set_title(f"Final losses vs coupling strength (N={N}, T={T}, L={L})")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"bath_sweep_N{N}_T{T}_L{L}_final_vs_g.png"), dpi=150)
print(f"saved Figures/bath_sweep_N{N}_T{T}_L{L}_final_vs_g.png")
