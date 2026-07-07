"""Measure per-worker wall time and peak RSS for 1 training epoch of the
MPO-building (use_compressed=False) path, so we can size the SLURM job."""
import os, sys, time, resource
os.environ.setdefault("OMP_NUM_THREADS", sys.argv[1] if len(sys.argv) > 1 else "16")
os.environ["OMP_NUM_THREADS"] = sys.argv[1] if len(sys.argv) > 1 else "16"
os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
threads = int(os.environ["OMP_NUM_THREADS"])
import torch
torch.set_num_threads(threads)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
for p in (PARENT_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
import TDME_Trott as tdme
from Learning_unitary.load_training_data import load_mps
import json

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

N, T = 30, 3
gamma_name = 2
gamma = [gamma_name * 0.01] * N
data_dir = os.path.join(SCRIPT_DIR, "Learning_data",
                        f"N{N}_T30_mu1.0_gamma{round(gamma[0],2)}_t0.2")
with open(os.path.join(data_dir, "params.json")) as f:
    n = json.load(f)["num_samples"]
MPS_weight1 = [load_mps(os.path.join(data_dir, f"input_MPS_{i}.npz")) for i in range(n)]
target = [load_mps(os.path.join(data_dir, f"target_MPS_{i}.npz"), input=False) for i in range(n)]
for m in MPS_weight1:
    for t in m.tensors: t.data.requires_grad_(False)
for m in target:
    for t in m.tensors: t.data.requires_grad_(False)
MPS_weight1 = [m.astype("complex128") for m in MPS_weight1]
target = [m.astype("complex128") for m in target]

print(f"[threads={threads}] loaded {n} samples  RSS_after_load={rss_mb():.0f} MB", flush=True)

model = tdme.QMLM(N, T)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(2):
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for i in range(n):
        mpo_fit = model.get_MPO(noise_type="dephasing").astype("complex128")
        M_input = mpo_fit | MPS_weight1[i]
        loss_i = tdme.tensor_network_distance(M_input, target[i], normalized=False)
        (loss_i / n).backward()
        running += loss_i.detach().item()
    optimizer.step()
    print(f"[threads={threads}] epoch {epoch}: loss={running/n:.4f}  wall={time.time()-t0:.1f}s  RSS={rss_mb():.0f} MB", flush=True)
