"""Test the use_compressed=True training path for QMLM-with-bath.

Compares three configurations on the same teacher / student / inputs:

  (A) use_compressed=False                                  — baseline MPO-build path
  (B) use_compressed=True, truncation=False (current scheduler default)
                                                            — layer-by-layer, no bond cap
  (C) use_compressed=True, truncation=True                  — layer-by-layer, bond cap at max_bd

Checks:
  - Forward losses A vs B should be identical (no truncation in either; only the
    contraction order differs).
  - C should be close to A/B for max_bd large enough relative to the student's
    natural bond.
  - Backward should produce finite gradients in all three (no NaNs).
  - Per-epoch wallclock should drop sharply going from A → B → C as N, T grow.

Uses small N, T_L to fit comfortably in a debugjob.
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
import torch.optim as optim

import TDME_Trott as tdme
from TDME_Trott import (
    QMLM, get_input_and_output_MPS_with_bath, tensor_network_distance,
    apply_mpo_to_mps_compressed, _strip_size1_outer_bonds,
)


def build_model_and_teacher(N, T_L, g, max_bd, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    MPS_weight1, target_mps_list, haar_list, weak_list = get_input_and_output_MPS_with_bath(
        N, T_L, coupling_strength=g, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6, num_threads=1,
    )
    MPS_weight1     = [m.astype("complex128") for m in MPS_weight1]
    target_mps_list = [m.astype("complex128") for m in target_mps_list]

    # student
    p_depolar_MPO = torch.ones((T_L, N), dtype=torch.float64) * 0.05
    rand_param = (
        torch.rand(T_L, N, 16, dtype=torch.float64)
        + 1j * torch.rand(T_L, N, 16, dtype=torch.float64)
    )
    rand_param = torch.nn.Parameter(rand_param, requires_grad=True)
    model = QMLM(N, T_L, param=rand_param, p_depolar=p_depolar_MPO)
    return model, MPS_weight1, target_mps_list


def epoch_path_A(model, inputs, targets):
    """Baseline MPO-build path."""
    losses = []
    mpo_fit = model.get_MPO(noise_type="depolarizing")
    for i, mps_in in enumerate(inputs):
        joined = mpo_fit | mps_in
        l = tensor_network_distance(joined.astype("complex128"), targets[i])
        losses.append(l)
    return torch.stack(losses).mean()


def epoch_path_B(model, inputs, targets, max_bd, max_err):
    """Layer-by-layer, no bond cap (current scheduler call signature)."""
    output_list = apply_mpo_to_mps_compressed(
        model, inputs, max_bond=max_bd, cutoff=max_err, noise_type="depolarizing",
    )
    losses = []
    for i, M_out in enumerate(output_list):
        out_stripped = _strip_size1_outer_bonds(M_out)
        l = tensor_network_distance(out_stripped.astype("complex128"), targets[i])
        losses.append(l)
    return torch.stack(losses).mean()


def epoch_path_C(model, inputs, targets, max_bd, max_err):
    """Layer-by-layer, with truncation=True (intended fast path)."""
    output_list = apply_mpo_to_mps_compressed(
        model, inputs, max_bond=max_bd, cutoff=max_err,
        noise_type="depolarizing", truncation=True,
    )
    losses = []
    for i, M_out in enumerate(output_list):
        out_stripped = _strip_size1_outer_bonds(M_out)
        l = tensor_network_distance(out_stripped.astype("complex128"), targets[i])
        losses.append(l)
    return torch.stack(losses).mean()


def time_path(name, fn, model, inputs, targets, n_iter=3):
    """Time forward+backward over n_iter iterations. Returns (mean_loss, mean_time_s)."""
    times, losses = [], []
    optimizer = optim.Adam(model.parameters(), lr=0.0)  # lr=0 so we don't perturb params
    for it in range(n_iter):
        t0 = time.time()
        loss = fn(model, inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # optimizer.step()   # skip: lr=0 anyway, and we want repeatable losses
        dt = time.time() - t0
        times.append(dt)
        losses.append(loss.item())
        print(f"    iter {it}: loss={loss.item():.6e}  dt={dt:.2f}s", flush=True)
    return float(np.mean(losses)), float(np.median(times))


def main():
    N      = int(os.environ.get("TEST_N", 10))
    T_L    = int(os.environ.get("TEST_TL", 3))
    g      = float(os.environ.get("TEST_G", 0.20))
    max_bd = int(os.environ.get("TEST_MAX_BD", 16))

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))))
    print(f"[test compressed]  N={N}  T=L={T_L}  g={g}  max_bd={max_bd}", flush=True)

    t0 = time.time()
    model, inputs, targets = build_model_and_teacher(N, T_L, g, max_bd, seed=100*T_L)
    print(f"  built teacher + student in {time.time()-t0:.1f}s", flush=True)

    print("\n=== Path A: use_compressed=False (MPO-build path) ===", flush=True)
    try:
        lA, tA = time_path("A", epoch_path_A, model, inputs, targets)
    except Exception as e:
        print(f"  ERROR in path A: {type(e).__name__}: {e}", flush=True)
        lA, tA = float('nan'), float('nan')

    print("\n=== Path B: use_compressed=True, truncation=False (current default) ===", flush=True)
    try:
        lB, tB = time_path("B", lambda m,i,t: epoch_path_B(m,i,t, max_bd, 1e-6),
                           model, inputs, targets)
    except Exception as e:
        print(f"  ERROR in path B: {type(e).__name__}: {e}", flush=True)
        lB, tB = float('nan'), float('nan')

    print("\n=== Path C: use_compressed=True, truncation=True (intended fast path) ===", flush=True)
    try:
        lC, tC = time_path("C", lambda m,i,t: epoch_path_C(m,i,t, max_bd, 1e-6),
                           model, inputs, targets)
    except Exception as e:
        print(f"  ERROR in path C: {type(e).__name__}: {e}", flush=True)
        lC, tC = float('nan'), float('nan')

    # Check gradients are finite for all paths
    def grad_norm(m):
        gs = [p.grad.detach() for p in m.parameters() if p.grad is not None]
        if not gs:
            return float('nan')
        flat = torch.cat([g.flatten().to(torch.complex128) for g in gs])
        return float(flat.abs().pow(2).sum().sqrt().item())

    print("\n=== Summary ===")
    print(f"  Path A loss = {lA:.6e}   per-iter = {tA:.2f}s")
    print(f"  Path B loss = {lB:.6e}   per-iter = {tB:.2f}s   (vs A: rel diff {abs(lB-lA)/max(abs(lA),1e-12):.3e})")
    print(f"  Path C loss = {lC:.6e}   per-iter = {tC:.2f}s   (vs A: rel diff {abs(lC-lA)/max(abs(lA),1e-12):.3e})")
    print(f"  grad norm after last backward = {grad_norm(model):.3e}", flush=True)


if __name__ == "__main__":
    main()
