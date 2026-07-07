"""Benchmark tensor_network_distance cost for the N=10, T=L=3/4/5 regime.

Reports per-sample timing for the `use_compressed=False` path that
Learning_QMLM_with_bath_scheduler uses. First call is warm-up (cotengra path
search); subsequent calls should hit the cache.
"""
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Learning_Lindbladian"))

import numpy as np
import torch

import TDME_Trott as tdme

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))))

N = 10
depol = 0.1
coupling = 0.1

for T_L in [3, 4, 5]:
    print(f"\n{'='*60}\nN={N}, T=L={T_L}\n{'='*60}", flush=True)

    # Teacher + targets
    t0 = time.time()
    MPS_weight1, target_mps_list, haar_list, weak_list = (
        tdme.get_input_and_output_MPS_with_bath(
            N, T_L, coupling_strength=coupling, J_b=1.0,
            truncation=True, max_bd=16, max_err=1e-6, num_threads=1,
        )
    )
    print(f"  teacher data gen (18 inputs, 3N=30 samples): {time.time()-t0:.2f}s", flush=True)

    MPS_weight1 = [m.astype("complex128") for m in MPS_weight1]
    target_mps_list = [m.astype("complex128") for m in target_mps_list]

    # Student
    p_depolar = torch.ones((T_L, N), dtype=torch.float64) * depol * 0.5
    rand_param = (torch.rand(T_L, N, 16, dtype=torch.float64)
                  + 1j * torch.rand(T_L, N, 16, dtype=torch.float64))
    rand_param = torch.nn.Parameter(rand_param, requires_grad=True)
    model = tdme.QMLM(N, T_L, param=rand_param, p_depolar=p_depolar)

    t0 = time.time()
    mpo_fit = model.get_MPO(noise_type="depolarizing")
    print(f"  build_mpo: {time.time()-t0:.2f}s  (tensors={len(mpo_fit.tensors)})", flush=True)

    # Warm-up: single distance call (triggers cotengra path search)
    t0 = time.time()
    M_input = mpo_fit | MPS_weight1[0]
    loss = tdme.tensor_network_distance(
        M_input.astype("complex128"), target_mps_list[0], normalized=False,
    )
    warm = time.time() - t0
    print(f"  1st tensor_network_distance (warm-up, includes path search): {warm:.2f}s  loss={loss.item():.4f}",
          flush=True)

    # 5 timed distance calls (path-cache hit)
    t_dist = []
    for s in range(5):
        t0 = time.time()
        M_in = mpo_fit | MPS_weight1[s]
        loss = tdme.tensor_network_distance(
            M_in.astype("complex128"), target_mps_list[s], normalized=False,
        )
        t_dist.append(time.time() - t0)
    print(f"  5 subsequent distance calls (cached): "
          f"mean={np.mean(t_dist):.2f}s  min={min(t_dist):.2f}s  max={max(t_dist):.2f}s",
          flush=True)

    # Full forward pass over all 30 samples + sum (one epoch's forward work)
    t0 = time.time()
    losses = []
    for sample_idx in range(len(MPS_weight1)):
        M_in = mpo_fit | MPS_weight1[sample_idx]
        losses.append(tdme.tensor_network_distance(
            M_in.astype("complex128"), target_mps_list[sample_idx], normalized=False,
        ))
    loss_epoch = torch.stack(losses).mean()
    fwd_time = time.time() - t0
    print(f"  full forward (30 samples -> stack -> mean): {fwd_time:.2f}s  loss={loss_epoch.item():.4f}",
          flush=True)

    # Backward pass timing
    t0 = time.time()
    loss_epoch.backward()
    bwd_time = time.time() - t0
    print(f"  backward (30 samples): {bwd_time:.2f}s", flush=True)

    # One-epoch time estimate
    epoch_est = fwd_time + bwd_time
    print(f"  ==> estimated per-epoch cost: {epoch_est:.1f}s  "
          f"(100 epochs ≈ {100*epoch_est/60:.1f} min)", flush=True)
