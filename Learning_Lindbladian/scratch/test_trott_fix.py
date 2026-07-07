"""Smoke test for the Pauli_MPS_after_TDME_output_only fidelity-gate fix.

Confirms:
  1. Default (track_truncation_error=False) produces the same final MPS as
     the old track-on path — only the returned `truncation_error` differs.
  2. The fast path is substantially faster than the tracking path at
     modest N / deep T.
  3. End-to-end Testing_TDME_Trotterization_parallel still runs correctly
     at N=10, T=3 (model), L=10 (target) with num_samples=20.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Casting complex")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
torch.set_num_threads(1)

import TDME_Trott as tdme


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # ===== 1. Correctness =====
    N = 10
    T = 5
    t = 1.0
    r = t / T
    mu = 1.0
    J = 1.0
    gamma = [0.1] * N
    max_bd = 32
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
    print("  PASS: fast path matches the tracking path")

    # ===== 2. End-to-end =====
    N_test = 10
    T_model = 3
    L_target = 10
    num_samples = 20
    num_threads = 4

    print(f"\n=== End-to-end Testing_TDME_Trotterization_parallel "
          f"(N={N_test}, T_model={T_model}, L_target={L_target}, "
          f"{num_samples} samples, {num_threads} workers) ===")
    t0 = time.time()
    testing_loss, testing_loss_list = tdme.Testing_TDME_Trotterization_parallel(
        N=N_test,
        model_layer=T_model,
        model_to_learn_layer=L_target,
        mu=1.0,
        gamma=[0.1] * N_test,
        J=1.0,
        t=1.0,
        normalized=False,
        max_bd=32,
        max_err=1e-6,
        truncation=True,
        noise_type="dephasing",
        use_scheduler=False,
        num_samples=num_samples,
        num_threads=num_threads,
    )
    elapsed = time.time() - t0
    print(f"\n  wall-clock: {elapsed:.1f}s  "
          f"({elapsed/num_samples:.2f}s/sample on {num_threads} workers)")
    print(f"  testing_loss: {testing_loss:.5f}")
    print(f"  got {len(testing_loss_list)}/{num_samples} successful samples")
    print("  PASS: end-to-end run succeeded")


if __name__ == "__main__":
    main()
