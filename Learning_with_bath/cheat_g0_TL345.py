"""Cheat-init the QMLM student at g=0 by copying the teacher's Haar gates
verbatim, set p_depolar=0, and report the resulting loss as the "g=0" entry
for TL=3, 4, 5. This is the architectural floor — what the model COULD reach
at g=0 if the optimizer didn't get stuck in a bad basin.

Saves outputs in the same format used by run_bath_sweep.py so the plot script
picks them up transparently:
  bath_sweep_N10_T{TL}_L{TL}_g000_p010_bd64_learning_loss.npy
  bath_sweep_N10_T{TL}_L{TL}_g000_p010_bd64_testing_loss.npy
  bath_sweep_N10_T{TL}_L{TL}_g000_p010_bd64_testing_loss_list.npy
  bath_sweep_N10_T{TL}_L{TL}_g000_p010_bd64_model_param.npy
  bath_sweep_N10_T{TL}_L{TL}_g000_p010_bd64_model_p_depolar.npy
"""
import os, sys, time, warnings
import resource
for _lim in (resource.RLIMIT_CPU, resource.RLIMIT_AS):
    try:
        s, h = resource.getrlimit(_lim)
        resource.setrlimit(_lim, (h, h))
    except (ValueError, OSError, resource.error):
        pass
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Learning_Lindbladian"))
NPY_DIR = os.path.join(SCRIPT_DIR, "npy_outputs")
os.makedirs(NPY_DIR, exist_ok=True)

import numpy as np
import torch
import TDME_Trott as tdme
from TDME_Trott import (
    QMLM,
    get_input_and_output_MPS_with_bath,
    get_random_input_output_MPS_with_bath,
    apply_mpo_to_mps_compressed,
    tensor_network_distance,
    _strip_size1_outer_bonds,
)


def build_cheating_params(N, L, haar_list):
    """Return (T=L, N, 16) complex128 params whose construct_all_SU4 image
    matches the teacher's Haar gates in the brickwall slots used by the
    student. Filler slots get random unitaries (they're never contracted)."""
    N_pairs_o = N // 2
    N_pairs_e = (N - 1) // 2
    params = torch.zeros((L, N, 16), dtype=torch.complex128)
    for l in range(L):
        haar_l = haar_list[l]
        assert len(haar_l) == N_pairs_o + N_pairs_e
        # odd brick → student slots 0..N_pairs_o-1
        for k in range(N_pairs_o):
            params[l, k] = haar_l[k].to(torch.complex128).reshape(16)
        # even brick → student slots N//2..N//2+N_pairs_e-1
        for k in range(N_pairs_e):
            params[l, N // 2 + k] = haar_l[N_pairs_o + k].to(torch.complex128).reshape(16)
        for k in range(N_pairs_o + N_pairs_e, N):
            params[l, k] = tdme.Haar_random_unitary(4).to(torch.complex128).reshape(16)
    return params


def cheat_run(N, T_L, max_bd=64, n_test=100):
    g = 0.0
    seed = 100 * T_L
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))))

    print(f"\n=== cheat g=0  N={N}  T=L={T_L}  max_bd={max_bd} ===", flush=True)
    t0 = time.time()
    MPS_weight1, target_mps_list, haar_list, weak_list = get_input_and_output_MPS_with_bath(
        N, T_L, coupling_strength=g, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6, num_threads=1,
    )
    print(f"  teacher gen: {time.time()-t0:.1f}s", flush=True)
    MPS_weight1     = [m.astype("complex128") for m in MPS_weight1]
    target_mps_list = [m.astype("complex128") for m in target_mps_list]

    params = build_cheating_params(N, T_L, haar_list)
    # QR round-trip sanity
    all_U = tdme.construct_all_SU4(N, T_L, params)
    err = 0.0
    N_pairs_o = N // 2
    N_pairs_e = (N - 1) // 2
    for l in range(T_L):
        for k in range(N_pairs_o):
            err = max(err, (all_U[l, k] - haar_list[l][k].to(torch.complex128)).abs().max().item())
        for k in range(N_pairs_e):
            err = max(err, (all_U[l, N // 2 + k] - haar_list[l][N_pairs_o + k].to(torch.complex128)).abs().max().item())
    print(f"  QR round-trip max|U_built - U_teacher| = {err:.3e}", flush=True)

    p_depolar = torch.zeros((T_L, N), dtype=torch.float64)
    model = QMLM(N, T_L,
                 param=torch.nn.Parameter(params, requires_grad=False),
                 p_depolar=p_depolar)

    # Training-set loss (weight-1 inputs, 3N = 30 samples for N=10)
    t0 = time.time()
    with torch.no_grad():
        out_list = apply_mpo_to_mps_compressed(
            model, MPS_weight1, max_bond=max_bd, cutoff=1e-6,
            noise_type="depolarizing", truncation=True,
        )
        train_losses = []
        for i, M_out in enumerate(out_list):
            out_stripped = _strip_size1_outer_bonds(M_out)
            l = tensor_network_distance(out_stripped.astype("complex128"), target_mps_list[i]).item()
            train_losses.append(l)
    train_losses = np.asarray(train_losses)
    train_loss_mean = float(train_losses.mean())
    print(f"  train  (weight-1, n={len(train_losses)})  mean={train_loss_mean:.4e}  "
          f"max={train_losses.max():.4e}  min={train_losses.min():.4e}  ({time.time()-t0:.1f}s)",
          flush=True)

    # Save train artifacts immediately so a kill during the test block doesn't
    # destroy them. Test-loss file gets re-saved after the test block below.
    prefix_save = f"bath_sweep_N{N}_T{T_L}_L{T_L}_g000_p010_bd{max_bd}"
    np.save(os.path.join(NPY_DIR, f"{prefix_save}_learning_loss.npy"),
            np.asarray([train_loss_mean]))
    np.save(os.path.join(NPY_DIR, f"{prefix_save}_model_param.npy"),
            model.params.detach().numpy())
    np.save(os.path.join(NPY_DIR, f"{prefix_save}_model_p_depolar.npy"),
            model.p_depolar.detach().numpy())
    print(f"  [partial save] wrote {prefix_save}_{{learning_loss,model_param,model_p_depolar}}.npy",
          flush=True)

    # Test set: 100 random Pauli inputs through the same teacher
    t0 = time.time()
    test_data, test_target, _, _ = get_random_input_output_MPS_with_bath(
        N, T_L, no_sample=n_test,
        haar_list=haar_list, weak_list=weak_list,
        coupling_strength=g, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6,
    )
    test_data   = [m.astype("complex128") for m in test_data]
    test_target = [m.astype("complex128") for m in test_target]
    print(f"  test teacher gen ({n_test} samples): {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    with torch.no_grad():
        test_out_list = apply_mpo_to_mps_compressed(
            model, test_data, max_bond=max_bd, cutoff=1e-6,
            noise_type="depolarizing", truncation=True,
        )
        test_losses = []
        for i, M_out in enumerate(test_out_list):
            out_stripped = _strip_size1_outer_bonds(M_out)
            l = tensor_network_distance(out_stripped.astype("complex128"), test_target[i]).item()
            test_losses.append(l)
    test_losses = np.asarray(test_losses)
    test_loss_mean = float(test_losses.mean())
    print(f"  test   (random, n={n_test})  mean={test_loss_mean:.4e}  "
          f"max={test_losses.max():.4e}  min={test_losses.min():.4e}  ({time.time()-t0:.1f}s)",
          flush=True)

    # Save in run_bath_sweep.py format
    prefix = f"bath_sweep_N{N}_T{T_L}_L{T_L}_g000_p010_bd{max_bd}"
    # learning_loss: store the (constant) train loss as a 1-element array so
    # plot_bath_sweep.py's ll[-1] read picks up the right value.
    np.save(os.path.join(NPY_DIR, f"{prefix}_learning_loss.npy"),
            np.asarray([train_loss_mean]))
    np.save(os.path.join(NPY_DIR, f"{prefix}_testing_loss.npy"),
            np.asarray([test_loss_mean]))
    np.save(os.path.join(NPY_DIR, f"{prefix}_testing_loss_list.npy"), test_losses)
    np.save(os.path.join(NPY_DIR, f"{prefix}_model_param.npy"),
            model.params.detach().numpy())
    np.save(os.path.join(NPY_DIR, f"{prefix}_model_p_depolar.npy"),
            model.p_depolar.detach().numpy())
    print(f"  saved {prefix}_*.npy", flush=True)
    return train_loss_mean, test_loss_mean


def main():
    N      = int(os.environ.get("CHEAT_N", 10))
    max_bd = int(os.environ.get("CHEAT_MAX_BD", 64))
    n_test = int(os.environ.get("CHEAT_N_TEST", 30))
    # Restrict to a single T_L via env (e.g. CHEAT_TL=4) so each debugjob fits
    # the 5400 cpu-sec/process rlimit (teacher gen for the test block dominates).
    tl_env = os.environ.get("CHEAT_TL", "")
    if tl_env:
        T_LS = [int(x) for x in tl_env.split(",")]
    else:
        T_LS = [3, 4, 5]
    results = {}
    for T_L in T_LS:
        results[T_L] = cheat_run(N, T_L, max_bd=max_bd, n_test=n_test)
    print("\n=== Summary ===")
    print(f"{'T=L':>4}  {'train':>12}  {'test':>12}")
    for T_L, (tr, te) in results.items():
        print(f"{T_L:>4}  {tr:>12.4e}  {te:>12.4e}")


if __name__ == "__main__":
    main()
