"""N=15 speedup check for the Pauli_MPS_after_TDME_output_only fidelity-gate fix."""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Casting complex")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

import TDME_Trott as tdme


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    N = 12
    T = 5
    t = 0.8
    r = t / T
    mu = 1.0
    J = 1.0
    gamma = [0.1] * N
    max_bd = 64
    max_err = 1e-8

    all_unitary = tdme.construct_TDME_unitary(N, T, r=r, mu=mu, J=J)
    all_jumping = tdme.construct_jump_matrices(N, gamma, r=r)

    torch.manual_seed(42); np.random.seed(42)
    M_input_fast, _ = tdme.random_pauli_MPS(N)
    torch.manual_seed(42); np.random.seed(42)
    M_input_track, _ = tdme.random_pauli_MPS(N)

    t0 = time.time()
    M_fast, te_fast = tdme.Pauli_MPS_after_TDME_output_only(
        M_input_fast.copy(), T, r=r,
        all_unitary=all_unitary, all_jumping=all_jumping,
        max_bd=max_bd, max_err=max_err,
        truncation=True, track_truncation_error=False,
    )
    t_fast = time.time() - t0

    t0 = time.time()
    M_track, te_track = tdme.Pauli_MPS_after_TDME_output_only(
        M_input_track.copy(), T, r=r,
        all_unitary=all_unitary, all_jumping=all_jumping,
        max_bd=max_bd, max_err=max_err,
        truncation=True, track_truncation_error=True,
    )
    t_track = time.time() - t0

    d = tdme.tensor_network_distance(
        M_fast.astype("complex128"), M_track.astype("complex128")
    ).item()

    print(f"\n=== Correctness (N={N}, T={T}, max_bd={max_bd}) ===")
    print(f"  fast path wall-clock   : {t_fast:.2f}s   te={te_fast}")
    print(f"  track path wall-clock  : {t_track:.2f}s   te={te_track:.3e}")
    print(f"  ||M_fast - M_track||^2 : {d:.3e}")
    print(f"  speedup (track / fast) : {t_track / max(t_fast, 1e-9):.2f}x")
    assert d < 1e-10, f"fast vs track mismatch: distance {d}"
    print("  PASS")


if __name__ == "__main__":
    main()
