#!/bin/bash
# debugjob 1 ./test_tl2_bd256_timing.sh
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
source /home/simba/.virtualenvs/QIP/bin/activate
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export GOMP_CPU_AFFINITY="0-31"
export OMP_PROC_BIND=close
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"
taskset -c 0-31 python -u - <<'PY'
import os, sys, time, warnings
warnings.filterwarnings("ignore", message="Casting complex")
SCRIPT_DIR = "/scratch/simba/Learning_with_bath"
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Learning_Lindbladian"))
import numpy as np, torch, torch.optim as optim
import TDME_Trott as tdme
from TDME_Trott import (
    QMLM, get_input_and_output_MPS_with_bath, tensor_network_distance,
    apply_mpo_to_mps_compressed, _strip_size1_outer_bonds,
)
N, T_L, g, max_bd = 10, 2, 0.20, 256
n_epochs = 5
torch.manual_seed(200); np.random.seed(200)
torch.set_num_threads(32)
print(f"[timing] N={N} T_L={T_L} g={g} max_bd={max_bd} cores=32 epochs={n_epochs}", flush=True)
t0=time.time()
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
optimizer = optim.Adam(model.parameters(), lr=0.05, betas=(0.9,0.999), amsgrad=True)
for epoch in range(n_epochs):
    t0=time.time()
    output_mps_list = apply_mpo_to_mps_compressed(
        model, MPS_weight1, max_bond=max_bd, cutoff=1e-6,
        noise_type="depolarizing", truncation=True,
    )
    losses = []
    for i, M_out in enumerate(output_mps_list):
        out_stripped = _strip_size1_outer_bonds(M_out)
        l = tensor_network_distance(out_stripped.astype("complex128"), target_mps_list[i])
        losses.append(l)
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    dt = time.time()-t0
    print(f"  epoch {epoch}: loss={loss.item():.4e} dt={dt:.2f}s", flush=True)
PY
