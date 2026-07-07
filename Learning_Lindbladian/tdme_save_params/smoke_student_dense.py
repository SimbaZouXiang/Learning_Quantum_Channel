"""Smoke: materialize one trained student channel as a 4^N x 4^N dense matrix.

For N=8 this is 65536 x 65536 float64 = 32 GB. Measure build time and memory.
"""
import os, sys, time, resource
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, str(threads))
os.environ.setdefault("COTENGRA_PARALLEL", "false")

import torch
torch.set_num_threads(threads)
import TDME_Trott as tdme

N        = 8
T        = 3
L_TARGET = 10
T_TIME   = 1.0
GAMMA    = 0.1
VARIANT  = "w1"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def main():
    prefix = os.path.join(SCRIPT_DIR, "results",
        f"N{N}_T{T}_L{L_TARGET}_t{T_TIME}_g{int(round(GAMMA*100)):03d}_{VARIANT}")
    print(f"Loading cell: {os.path.basename(prefix)}", flush=True)
    state = torch.load(prefix + "_state_dict.pt", map_location="cpu", weights_only=True)
    model = tdme.QMLM(N, T)
    model.load_state_dict(state)
    model.eval()
    print(f"  loaded.  rss={rss_gb():.2f} GB", flush=True)

    t0 = time.time()
    mpo = model.get_MPO(noise_type="dephasing")
    print(f"  built MPO in {time.time()-t0:.1f}s.  rss={rss_gb():.2f} GB", flush=True)

    # cast every tensor to a uniform dtype (complex128) before contracting.
    # PTensors need to be unparametrized first.
    import quimb.tensor as qtn
    new_tensors = []
    for t in mpo.tensors:
        plain = t.unparametrize() if hasattr(t, "unparametrize") else t.copy()
        if torch.is_tensor(plain.data):
            plain.modify(data=plain.data.to(torch.complex128))
        else:
            plain.modify(data=np.asarray(plain.data).astype(np.complex128))
        new_tensors.append(plain)
    mpo_plain = qtn.TensorNetwork(new_tensors)
    print(f"  cast to complex128.  rss={rss_gb():.2f} GB", flush=True)

    t0 = time.time()
    M = mpo_plain.to_dense([f'k{i}' for i in range(N)],
                            [f'input{i}' for i in range(N)])
    elapsed = time.time() - t0
    print(f"  materialized dense in {elapsed:.1f}s", flush=True)
    if torch.is_tensor(M):
        M_np = M.detach().cpu().numpy()
    else:
        M_np = np.asarray(M)
    print(f"  shape={tuple(M_np.shape)}  dtype={M_np.dtype}  "
          f"nbytes={M_np.nbytes / 1024**3:.2f} GB", flush=True)
    print(f"  rss={rss_gb():.2f} GB  (peak)", flush=True)
    max_im = float(np.max(np.abs(M_np.imag)))
    print(f"  max |imag part|: {max_im:.2e}", flush=True)
    if max_im < 1e-10:
        M_real = np.ascontiguousarray(M_np.real)
        del M_np
        print(f"  reduced to float64.  nbytes={M_real.nbytes/1024**3:.2f} GB", flush=True)
        print(f"  Frobenius norm: {np.linalg.norm(M_real):.4f}", flush=True)
        print(f"  trace: {np.trace(M_real):.4f}", flush=True)
        print(f"  max |entry|: {np.max(np.abs(M_real)):.4f}", flush=True)
        out = prefix + "_dense.npy"
        np.save(out, M_real)
        print(f"  saved {out} ({os.path.getsize(out)/1024**3:.2f} GB)", flush=True)
    else:
        print(f"  KEEPING complex (imaginary part is non-trivial)", flush=True)
        print(f"  Frobenius norm: {np.linalg.norm(M_np):.4f}", flush=True)


if __name__ == "__main__":
    main()
