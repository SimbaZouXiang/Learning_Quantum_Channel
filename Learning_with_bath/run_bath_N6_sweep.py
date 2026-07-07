"""Coupling-strength sweep for QMLM-with-bath at N=6, T=3, L=6.

Runs one coupling strength per invocation, selected from COUPLINGS[index],
where index comes from $SLURM_ARRAY_TASK_ID. Each run saves its own .npy files.

Companion plotting script (`plot_bath_N6_sweep.py`) loads all sweep .npy files
and produces:
  - loss curves overlay (one line per coupling strength)
  - final train/test loss vs. coupling strength
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Learning_Lindbladian"))
NPY_DIR = os.path.join(SCRIPT_DIR, "npy_outputs")
os.makedirs(NPY_DIR, exist_ok=True)

import numpy as np
import torch

import TDME_Trott as tdme

COUPLINGS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def main():
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if idx_str is None:
        raise SystemExit("SLURM_ARRAY_TASK_ID not set; run as a SLURM array job.")
    idx = int(idx_str)
    assert 0 <= idx < len(COUPLINGS), f"index {idx} out of range"
    coupling_strength = COUPLINGS[idx]

    # Fixed seed per task index so the Haar unitaries are the *same* across
    # couplings (apples-to-apples: only the weak gate changes).
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 4)))

    N = 6
    T = 3              # student layers
    L = 6              # teacher layers
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
    print(f"=== g={coupling_strength}: finished in {elapsed:.1f}s ===", flush=True)

    (model, learning_loss, haar_list, weak_list,
     testing_loss, testing_loss_list,
     model_param, model_p_depolar,
     model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ) = out

    g_tag = f"{int(round(coupling_strength * 100)):03d}"
    prefix = (f"bath_sweep_N{N}_T{T}_L{L}_g{g_tag}"
              f"_p{int(depolarizing_strength*100):03d}")
    np.save(os.path.join(NPY_DIR, f"{prefix}_learning_loss.npy"), np.array(learning_loss))
    np.save(os.path.join(NPY_DIR, f"{prefix}_testing_loss.npy"), np.array([testing_loss]))
    np.save(os.path.join(NPY_DIR, f"{prefix}_testing_loss_list.npy"), np.array(testing_loss_list))
    np.save(os.path.join(NPY_DIR, f"{prefix}_model_param.npy"), model_param)
    np.save(os.path.join(NPY_DIR, f"{prefix}_model_p_depolar.npy"), model_p_depolar)
    print(f"saved {prefix}_*.npy to {NPY_DIR}", flush=True)


if __name__ == "__main__":
    main()
