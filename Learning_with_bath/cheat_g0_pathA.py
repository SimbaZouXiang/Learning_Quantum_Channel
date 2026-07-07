"""Path-A cheat-init comparison for TL=4 / TL=5 at g=0, max_bd=256.

Uses use_compressed=False (the MPO-build path) so the student is computed
exactly with no bond truncation. The remaining loss should reflect only the
teacher's target-side truncation, which is the lower bound for Path C.

Saves NOTHING — this is a one-shot diagnostic comparison.
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
import numpy as np, torch
import TDME_Trott as tdme
from TDME_Trott import (
    QMLM, get_input_and_output_MPS_with_bath, get_random_input_output_MPS_with_bath,
    tensor_network_distance, _strip_size1_outer_bonds,
)
from cheat_g0_TL345 import build_cheating_params

def path_A_loss(model, input_mps_list, target_mps_list):
    losses = []
    mpo_fit = model.get_MPO(noise_type="depolarizing")
    with torch.no_grad():
        for i, mps_in in enumerate(input_mps_list):
            joined = mpo_fit | mps_in
            l = tensor_network_distance(joined.astype("complex128"), target_mps_list[i]).item()
            losses.append(l)
    return np.asarray(losses)

def run(T_L, n_test=10, max_bd=256):
    N = 10
    g = 0.0
    seed = 100 * T_L
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", 32))))
    print(f"\n=== Path A cheat g=0  N={N}  T_L={T_L}  max_bd={max_bd}  n_test={n_test} ===", flush=True)

    t0 = time.time()
    MPS_w1, target_w1, haar_list, weak_list = get_input_and_output_MPS_with_bath(
        N, T_L, coupling_strength=g, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6, num_threads=1,
    )
    print(f"  teacher gen (weight-1): {time.time()-t0:.1f}s", flush=True)
    MPS_w1     = [m.astype("complex128") for m in MPS_w1]
    target_w1  = [m.astype("complex128") for m in target_w1]

    params = build_cheating_params(N, T_L, haar_list)
    p_depolar = torch.zeros((T_L, N), dtype=torch.float64)
    model = QMLM(N, T_L,
                 param=torch.nn.Parameter(params, requires_grad=False),
                 p_depolar=p_depolar)

    t0 = time.time()
    train_losses = path_A_loss(model, MPS_w1, target_w1)
    dt = time.time() - t0
    print(f"  Path A train (weight-1, n={len(train_losses)})  mean={train_losses.mean():.4e}  "
          f"max={train_losses.max():.4e}  ({dt:.1f}s, {dt/len(train_losses):.2f}s/sample)", flush=True)

    t0 = time.time()
    test_data, test_target, _, _ = get_random_input_output_MPS_with_bath(
        N, T_L, no_sample=n_test,
        haar_list=haar_list, weak_list=weak_list,
        coupling_strength=g, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6,
    )
    test_data   = [m.astype("complex128") for m in test_data]
    test_target = [m.astype("complex128") for m in test_target]
    print(f"  teacher gen (test, {n_test}): {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    test_losses = path_A_loss(model, test_data, test_target)
    dt = time.time() - t0
    print(f"  Path A test  (random, n={n_test})    mean={test_losses.mean():.4e}  "
          f"max={test_losses.max():.4e}  ({dt:.1f}s, {dt/n_test:.2f}s/sample)", flush=True)
    return train_losses.mean(), test_losses.mean()

def main():
    T_L    = int(os.environ.get("PATHA_TL", 4))
    n_test = int(os.environ.get("PATHA_N_TEST", 10))
    max_bd = int(os.environ.get("PATHA_MAX_BD", 256))
    tr, te = run(T_L, n_test=n_test, max_bd=max_bd)
    print(f"\n=== Summary  TL={T_L}  bd={max_bd}  Path A ===")
    print(f"  train (weight-1): {tr:.4e}")
    print(f"  test  (random):   {te:.4e}")

if __name__ == "__main__":
    main()
