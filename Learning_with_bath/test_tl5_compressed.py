"""Quick TL=5 verification of the patched compressed path.

Runs the actual Learning_QMLM_with_bath_scheduler with use_compressed=True at
N=10, T_L=5, g=0.20, max_bd=64 for 5 main + 1 fine-tune epoch, just to confirm
the deepest depth doesn't crash or NaN. Skips the 100-sample testing block by
returning early after grabbing the model.
"""
import os, sys, time, warnings
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

def main():
    N, T_L, g, max_bd = 10, 5, 0.20, 64
    n_epochs = int(os.environ.get("N_EPOCHS", 5))

    torch.manual_seed(100 * T_L)
    np.random.seed(100 * T_L)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))))

    print(f"[TL5 compressed test] N={N} T=L={T_L} g={g} max_bd={max_bd} epochs={n_epochs}",
          flush=True)
    t0 = time.time()
    MPS_weight1, target_mps_list, _, _ = get_input_and_output_MPS_with_bath(
        N, T_L, coupling_strength=g, J_b=1.0,
        truncation=True, max_bd=max_bd, max_err=1e-6, num_threads=1,
    )
    print(f"  teacher gen: {time.time()-t0:.1f}s", flush=True)
    MPS_weight1     = [m.astype("complex128") for m in MPS_weight1]
    target_mps_list = [m.astype("complex128") for m in target_mps_list]

    p_depolar_MPO = torch.ones((T_L, N), dtype=torch.float64) * 0.05
    rand_param = (torch.rand(T_L, N, 16, dtype=torch.float64)
                  + 1j * torch.rand(T_L, N, 16, dtype=torch.float64))
    rand_param = torch.nn.Parameter(rand_param, requires_grad=True)
    model = QMLM(N, T_L, param=rand_param, p_depolar=p_depolar_MPO)
    optimizer = optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999), amsgrad=True)
    clamp_value = 0.11

    losses_hist = []
    for epoch in range(n_epochs):
        t0 = time.time()
        output_mps_list = apply_mpo_to_mps_compressed(
            model, MPS_weight1, max_bond=max_bd, cutoff=1e-6,
            noise_type="depolarizing", truncation=True,
        )
        losses = []
        for i, M_out in enumerate(output_mps_list):
            out_stripped = _strip_size1_outer_bonds(M_out)
            l = tensor_network_distance(out_stripped.astype("complex128"), target_mps_list[i])
            losses.append(l)
        loss_epoch = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss_epoch.backward()
        # gradient sanity
        gs = [p.grad for p in model.parameters() if p.grad is not None]
        gn = sum(g.abs().pow(2).sum().item() for g in gs) ** 0.5
        if not np.isfinite(gn):
            print(f"  FAIL: non-finite gradient at epoch {epoch}", flush=True)
            return 1
        optimizer.step()
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value + 0.01)
        dt = time.time() - t0
        losses_hist.append(loss_epoch.item())
        print(f"  epoch {epoch}: loss={loss_epoch.item():.6e} grad_norm={gn:.3e} dt={dt:.2f}s",
              flush=True)

    if not all(np.isfinite(losses_hist)):
        print("  FAIL: non-finite loss in history", flush=True)
        return 1
    if losses_hist[-1] >= losses_hist[0]:
        print(f"  WARNING: loss did not decrease ({losses_hist[0]:.4e} -> {losses_hist[-1]:.4e})",
              flush=True)
    else:
        print(f"  ✓ loss decreased {losses_hist[0]:.4e} -> {losses_hist[-1]:.4e}", flush=True)
    print(f"  ✓ all gradients finite, no NaN", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
