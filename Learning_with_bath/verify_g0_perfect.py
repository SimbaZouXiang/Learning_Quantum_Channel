"""Confirm the QMLM-with-bath architecture can reach zero loss at g=0.

At g=0 the bath decouples (weak unitary = I), so after tracing out the bath the
teacher's effect on the data qubits is exactly L brick-wall layers of 2-site
Haar SU(4) gates. The depolarizing-only student QMLM has the *same* layout
(T layers of 2-site SU(4) brickwall + per-site depolarizing noise). Setting
p_depolar = 0 and copying the teacher's Haar gates into the student should
therefore reproduce the target exactly, up to bond-truncation noise.

We "cheat" the parameterization: construct_SU4_from_input does QR on
reshape(params, 4,4). For unitary U, qr(U) = (U, I) (the diagonal of R is real
positive), so passing params = U.flatten() yields Q = U. We exploit that here.

Run inside a debugjob via job_verify_g0.sh.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Learning_Lindbladian"))

import numpy as np
import torch

import TDME_Trott as tdme
from TDME_Trott import (
    QMLM,
    get_input_and_output_MPS_with_bath,
    get_random_input_output_MPS_with_bath,
    tensor_network_distance,
)


def build_cheating_params(N, L, haar_list):
    """Return a (T=L, N, 16) complex128 tensor whose construct_all_SU4 image
    matches the teacher's Haar gates slot-for-slot.

    Slot layout used by Build_QMLM_MPO / Apply_two_site_layer:
      all_U[layer, 0:N//2]      → odd brick (sites 0,2,4,...; N//2 gates)
      all_U[layer, N//2:N]      → even brick (sites 1,3,5,...; (N-1)//2 gates,
                                  so the last slot is unused for even N)

    Teacher haar_list[layer] is laid out as
      [N_pairs_o odd-brick gates, N_pairs_e even-brick gates]
    matching the student exactly except for the unused tail slot.
    """
    N_pairs_o = N // 2
    N_pairs_e = (N - 1) // 2
    params = torch.zeros((L, N, 16), dtype=torch.complex128)
    for l in range(L):
        haar_l = haar_list[l]
        assert len(haar_l) == N_pairs_o + N_pairs_e
        # odd
        for k in range(N_pairs_o):
            U = haar_l[k].to(torch.complex128)
            params[l, k] = U.reshape(16)
        # even — student stores them at offset N//2
        for k in range(N_pairs_e):
            U = haar_l[N_pairs_o + k].to(torch.complex128)
            params[l, N // 2 + k] = U.reshape(16)
        # any unused slot (e.g. params[l, N-1] when N is even) gets a random
        # SU(4); construct_SU4_from_input produces a unitary, but it is never
        # contracted since Apply_two_site_layer's loop stops at N-1. Use a
        # Haar-random unitary so QR's diagonal is well conditioned.
        for k in range(N_pairs_o + N_pairs_e, N):
            U = tdme.Haar_random_unitary(4).to(torch.complex128)
            params[l, k] = U.reshape(16)
    return params


def round_trip_check(params, haar_list):
    """Sanity: construct_all_SU4(params) should yield exactly haar_list at the
    used slots. Returns (max_err_odd, max_err_even)."""
    L, N, _ = params.shape
    all_U = tdme.construct_all_SU4(N, L, params)
    N_pairs_o = N // 2
    N_pairs_e = (N - 1) // 2
    err_odd = 0.0
    err_even = 0.0
    for l in range(L):
        for k in range(N_pairs_o):
            U_target = haar_list[l][k].to(torch.complex128)
            err_odd = max(err_odd, (all_U[l, k] - U_target).abs().max().item())
        for k in range(N_pairs_e):
            U_target = haar_list[l][N_pairs_o + k].to(torch.complex128)
            err_even = max(err_even, (all_U[l, N // 2 + k] - U_target).abs().max().item())
    return err_odd, err_even


def main():
    N      = int(os.environ.get("VERIFY_N", 10))
    T_L    = int(os.environ.get("VERIFY_TL", 3))
    max_bd = int(os.environ.get("VERIFY_MAX_BD", 16))
    seed   = int(os.environ.get("VERIFY_SEED", 100 * T_L))

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))))

    print(f"[verify g=0]  N={N}  T=L={T_L}  max_bd={max_bd}  seed={seed}", flush=True)

    t0 = time.time()
    MPS_weight1, target_mps_list, haar_list, weak_list = get_input_and_output_MPS_with_bath(
        N, T_L,
        haar_list=None, weak_list=None,
        coupling_strength=0.0, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6,
        num_threads=1,
    )
    print(f"  built {len(MPS_weight1)} weight-1 inputs and targets in {time.time()-t0:.1f}s",
          flush=True)

    MPS_weight1     = [m.astype("complex128") for m in MPS_weight1]
    target_mps_list = [m.astype("complex128") for m in target_mps_list]

    params = build_cheating_params(N, T_L, haar_list)
    err_odd, err_even = round_trip_check(params, haar_list)
    print(f"  round-trip QR sanity:  max|U_constructed - U_teacher|"
          f"  odd={err_odd:.3e}  even={err_even:.3e}", flush=True)

    p_depolar = torch.zeros((T_L, N), dtype=torch.float64)
    model = QMLM(N, T_L,
                 param=torch.nn.Parameter(params, requires_grad=False),
                 p_depolar=p_depolar)

    t0 = time.time()
    losses = []
    with torch.no_grad():
        mpo_fit = model.get_MPO(noise_type="depolarizing")
        for i, mps_in in enumerate(MPS_weight1):
            M_out = mpo_fit | mps_in
            l = tensor_network_distance(
                M_out.astype("complex128"),
                target_mps_list[i],
            ).item()
            losses.append(l)
    losses = np.asarray(losses)
    print(f"  evaluated cheating student on weight-1 inputs in {time.time()-t0:.1f}s",
          flush=True)
    print(f"  weight-1 loss   mean={losses.mean():.3e}   max={losses.max():.3e}"
          f"   min={losses.min():.3e}", flush=True)

    # Random-Pauli test set, same teacher, to mirror the testing_loss field.
    test_data, test_target, _, _ = get_random_input_output_MPS_with_bath(
        N, T_L, no_sample=30,
        haar_list=haar_list, weak_list=weak_list,
        coupling_strength=0.0, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6,
    )
    test_data   = [m.astype("complex128") for m in test_data]
    test_target = [m.astype("complex128") for m in test_target]

    test_losses = []
    with torch.no_grad():
        mpo_fit = model.get_MPO(noise_type="depolarizing")
        for i, mps_in in enumerate(test_data):
            M_out = mpo_fit | mps_in
            l = tensor_network_distance(
                M_out.astype("complex128"),
                test_target[i],
            ).item()
            test_losses.append(l)
    test_losses = np.asarray(test_losses)
    print(f"  random-Pauli loss  mean={test_losses.mean():.3e}"
          f"  max={test_losses.max():.3e}  min={test_losses.min():.3e}", flush=True)

    np.save(os.path.join(SCRIPT_DIR, "npy_outputs",
                         f"verify_g0_N{N}_T{T_L}_L{T_L}_weight1_loss.npy"),
            losses)
    np.save(os.path.join(SCRIPT_DIR, "npy_outputs",
                         f"verify_g0_N{N}_T{T_L}_L{T_L}_test_loss.npy"),
            test_losses)
    print("  saved verify_g0_*.npy", flush=True)


if __name__ == "__main__":
    main()
