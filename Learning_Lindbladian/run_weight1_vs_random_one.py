"""A/B test: weight-1 Pauli training set vs random-Pauli training set, with
the same number of samples (3N) and the same training hyperparameters.

For one (N, L_target) pair selected by SLURM_ARRAY_TASK_ID, this:
  1. Generates a fixed random target QMLM(L) parameter tensor (seed depends
     only on (N, L)).
  2. Trains a 3-layer QMLM (T=3) student against that target, twice — once
     with weight-1 Pauli inputs, once with random Pauli inputs of weight in
     [1, N]. Identical optimizer/seed/epochs/etc.
  3. Saves the full `learning_loss` curve for each variant.

Outputs per task:
  W1RAND_N{N}_L{L}_T3_w1_learning_loss.npy
  W1RAND_N{N}_L{L}_T3_rand_learning_loss.npy
  W1RAND_N{N}_L{L}_T3_w1_testing_loss.npy
  W1RAND_N{N}_L{L}_T3_rand_testing_loss.npy

Grid: N ∈ {8, 10, 12} × L ∈ {3, 4, 5, 6, 7, 8} = 18 tasks (array index 0..17).
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Pin BLAS thread counts before importing torch.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(var, "8")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

import TDME_Trott as tdme

NS = [8, 10, 12]
LS = [3, 4, 5, 6, 7, 8]
T_MODEL = 3
EPOCHS = 100
LR = 0.05
DEPOLARIZING_STRENGTH = 0.2
MAX_BD = 32
MAX_ERR = 1e-6
TRUNCATION = True
NOISE_TYPE = "depolarizing"   # Learning_MPO_scheduler default path uses depolarizing-only
USE_COMPRESSED = False

OUT_DIR = os.path.join(SCRIPT_DIR, "Learning_result")
os.makedirs(OUT_DIR, exist_ok=True)


def _make_target_param(N, L):
    """Same param for both A and B runs."""
    g = torch.Generator().manual_seed(1000 * N + L)
    return (
        torch.rand(L, N, 16, dtype=torch.float64, generator=g)
        + 1j * torch.rand(L, N, 16, dtype=torch.float64, generator=g)
    )


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_and_save(N, L, weight_1, target_param):
    tag = "w1" if weight_1 else "rand"
    label = "weight-1" if weight_1 else "random"
    print(f"\n=== N={N} L={L} ({label}) ===", flush=True)
    _set_seed(20240501 + (0 if weight_1 else 1))

    t0 = time.time()
    (
        model, learning_loss, param_used, p_depolar_used,
        testing_loss, testing_loss_list,
        params_np, p_depolar_np,
        p_dephaseX_np, p_dephaseY_np, p_dephaseZ_np,
    ) = tdme.Learning_MPO_scheduler(
        N=N,
        MPO_layer=T_MODEL,
        model_to_learn_layer=L,
        param_list=target_param,
        depolarizing_strength=DEPOLARIZING_STRENGTH,
        epochs=EPOCHS,
        lr=LR,
        normalized=False,
        max_bd=MAX_BD,
        max_err=MAX_ERR,
        truncation=TRUNCATION,
        noise_type=NOISE_TYPE,
        use_compressed=USE_COMPRESSED,
        num_threads=1,
        weight_1_pauli_strings=weight_1,
    )
    elapsed = time.time() - t0

    prefix = os.path.join(OUT_DIR, f"W1RAND_N{N}_L{L}_T{T_MODEL}_{tag}")
    np.save(f"{prefix}_learning_loss.npy", np.array(learning_loss))
    np.save(f"{prefix}_testing_loss.npy", np.array(testing_loss))
    np.save(f"{prefix}_testing_loss_list.npy", np.array(testing_loss_list))
    print(f"  elapsed: {elapsed:.1f}s  final_train_loss={learning_loss[-1]:.5e}  "
          f"min_train_loss={min(learning_loss):.5e}  testing_loss={float(testing_loss):.5e}",
          flush=True)


def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= task_id < len(NS) * len(LS):
        raise ValueError(f"task_id {task_id} out of range")
    n_idx, l_idx = divmod(task_id, len(LS))
    N = NS[n_idx]
    L = LS[l_idx]
    print(f"[task {task_id}] N={N}, L={L}", flush=True)

    target_param = _make_target_param(N, L)
    _train_and_save(N, L, weight_1=True, target_param=target_param)
    _train_and_save(N, L, weight_1=False, target_param=target_param)
    print(f"\n[task {task_id}] DONE", flush=True)


if __name__ == "__main__":
    main()
