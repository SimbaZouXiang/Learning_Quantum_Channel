"""Time per-epoch training cost at different max_bd values, to extrapolate
total wall-clock for the bath sweep.

Runs 5 epochs of the actual training loop (forward + backward + Adam) for each
(T_L, max_bd) pair, prints per-epoch median, and extrapolates to 120 epochs.
Single setting per invocation via env vars.
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
)


def main():
    N      = int(os.environ.get("BENCH_N", 10))
    T_L    = int(os.environ.get("BENCH_TL", 5))
    max_bd = int(os.environ.get("BENCH_MAX_BD", 16))
    g      = float(os.environ.get("BENCH_G", 0.20))
    n_warm = int(os.environ.get("BENCH_WARM", 1))
    n_iter = int(os.environ.get("BENCH_ITER", 5))

    torch.manual_seed(100 * T_L)
    np.random.seed(100 * T_L)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))))

    print(f"[bench]  N={N}  T=L={T_L}  max_bd={max_bd}  g={g}  warm={n_warm}  iter={n_iter}",
          flush=True)

    t_gen = time.time()
    MPS_weight1, target_mps_list, _, _ = get_input_and_output_MPS_with_bath(
        N, T_L, coupling_strength=g, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6, num_threads=1,
    )
    t_gen = time.time() - t_gen
    print(f"  teacher generation: {t_gen:.1f}s", flush=True)

    MPS_weight1     = [m.astype("complex128") for m in MPS_weight1]
    target_mps_list = [m.astype("complex128") for m in target_mps_list]

    p_depolar_MPO = torch.ones((T_L, N), dtype=torch.float64) * 0.05
    rand_param = (
        torch.rand(T_L, N, 16, dtype=torch.float64)
        + 1j * torch.rand(T_L, N, 16, dtype=torch.float64)
    )
    rand_param = torch.nn.Parameter(rand_param, requires_grad=True)
    model = QMLM(N, T_L, param=rand_param, p_depolar=p_depolar_MPO)
    optimizer = optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999), amsgrad=True)
    clamp_value = min(0.1 * 1.1, 1.0)

    times = []
    for epoch in range(n_warm + n_iter):
        t0 = time.time()
        mpo_fit = model.get_MPO(noise_type="depolarizing")
        losses = []
        for i, mps_in in enumerate(MPS_weight1):
            M_input = mpo_fit | mps_in
            l = tensor_network_distance(M_input.astype("complex128"), target_mps_list[i])
            losses.append(l)
        loss_this_epoch = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss_this_epoch.backward()
        optimizer.step()
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value + 0.01)
        dt = time.time() - t0
        tag = "warm" if epoch < n_warm else "meas"
        print(f"  epoch {epoch:2d} [{tag}]  loss={loss_this_epoch.item():.4e}   dt={dt:.2f}s",
              flush=True)
        if epoch >= n_warm:
            times.append(dt)

    times = np.array(times)
    med = float(np.median(times))
    mean = float(np.mean(times))
    extrap = med * 120  # 100 main + 20 fine-tune
    print(f"  per-epoch  median={med:.2f}s  mean={mean:.2f}s",
          flush=True)
    print(f"  extrapolated training (120 epochs): {extrap/60:.1f} min "
          f"(+ {t_gen:.0f}s teacher gen)", flush=True)


if __name__ == "__main__":
    main()
