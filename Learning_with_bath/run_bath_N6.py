"""Training run for QMLM-with-bath at N=6, T=3, L=6, 100 epochs.

- N=6 data qubits; target has 2N=12 sites (d b d b d b d b d b d b).
- Teacher (no noise): L=6 layers of Haar(odd) + Haar(even) + weak(d-b).
- Student (with noise): T=3 layers of the regular QMLM.
- use_compressed=False because the pre-existing student compressed path
  fails for N>=4 with T>=2 (unrelated to the bath code).

Saves loss history and produces a plot.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Allow importing TDME_Trott from sibling `Learning_Lindbladian/` directory.
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Learning_Lindbladian"))
NPY_DIR = os.path.join(SCRIPT_DIR, "npy_outputs")
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 4)))

import TDME_Trott as tdme

N = 6
T = 3                  # student layers (QMLM, with noise)
L = 6                  # teacher layers (QMLM-with-bath, no noise)
coupling_strength = 0.05
J_b = 1.0
depolarizing_strength = 0.1
epochs = 100
lr = 0.05
max_bd = 16
max_err = 1e-6

t0 = time.time()
out = tdme.Learning_QMLM_with_bath_scheduler(
    N=N, MPO_layer=T, model_to_learn_layer=L,
    coupling_strength=coupling_strength, J_b=J_b,
    depolarizing_strength=depolarizing_strength,
    epochs=epochs, lr=lr,
    max_bd=max_bd, max_err=max_err, truncation=True,
    noise_type="depolarizing",
    use_compressed=False, num_threads=1,
)
elapsed = time.time() - t0
print(f"=== finished in {elapsed:.1f}s ===", flush=True)

(model, learning_loss, haar_list, weak_list,
 testing_loss, testing_loss_list,
 model_param, model_p_depolar,
 model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ) = out

prefix = (f"bath_N{N}_T{T}_L{L}_g{int(coupling_strength*100):03d}"
          f"_p{int(depolarizing_strength*100):03d}")
np.save(os.path.join(NPY_DIR, f"{prefix}_learning_loss.npy"), np.array(learning_loss))
np.save(os.path.join(NPY_DIR, f"{prefix}_testing_loss.npy"), np.array([testing_loss]))
np.save(os.path.join(NPY_DIR, f"{prefix}_testing_loss_list.npy"), np.array(testing_loss_list))
np.save(os.path.join(NPY_DIR, f"{prefix}_model_param.npy"), model_param)
np.save(os.path.join(NPY_DIR, f"{prefix}_model_p_depolar.npy"), model_p_depolar)
print(f"saved {prefix}_*.npy to {NPY_DIR}", flush=True)

# Plot loss curve.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

main_epochs = epochs
finetune_epochs = len(learning_loss) - main_epochs

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(learning_loss)), learning_loss, label="train loss", linewidth=1.6)
if finetune_epochs > 0:
    ax.axvspan(main_epochs, len(learning_loss), alpha=0.15, color="orange",
               label=f"fine-tune ({finetune_epochs} epochs)")
ax.axhline(testing_loss, linestyle="--", color="red",
           label=f"avg test loss = {testing_loss:.4f}")
ax.set_xlabel("epoch")
ax.set_ylabel("tensor_network_distance loss")
ax.set_title(f"QMLM-with-bath training: N={N}, T={T}, L={L}, g={coupling_strength}, "
             f"p_dep={depolarizing_strength}")
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
out_png = os.path.join(FIG_DIR, f"{prefix}_loss.png")
fig.savefig(out_png, dpi=150)
print(f"saved figure to {out_png}", flush=True)
