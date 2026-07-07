"""For each (t, gamma) cell, build dense PTMs for the teacher + 5 students,
compute pairwise trace (Schatten-1) norms and Frobenius distances, and append
to a CSV.

Trace norm of M = sum of singular values via numpy.linalg.svd(M, compute_uv=False).

Resumable: skips (t, gamma) cells already present in the CSV.
"""
import os, sys, time, gc, csv, itertools, json
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
import quimb.tensor as qtn
import TDME_Trott as tdme
import scipy.linalg as sla
import resource

from build_tdme_dense import build_tdme_dense


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

# -------------------- cell grid + constants --------------------
N        = 8
T_STU    = 3
L_TGT    = 10
MU       = 1
J_       = 1
NOISE    = "dephasing"

T_LIST     = [0.5, 1.0, 2.0, 3.0]
GAMMA_LIST = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50]
VARIANTS   = ["w1", "w2full", "combined", "random24", "random276"]
CHANNELS   = VARIANTS + ["target"]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OUT_CSV = os.path.join(RESULTS_DIR, "trace_norm_pairs.csv")
ONLY_T = os.environ.get("TRACE_T")
ONLY_G = os.environ.get("TRACE_GAMMA")
if ONLY_T:
    T_LIST = [float(x) for x in ONLY_T.split(",")]
if ONLY_G:
    GAMMA_LIST = [float(x) for x in ONLY_G.split(",")]


def _student_prefix(t, g, var):
    g_int = int(round(g * 100))
    return os.path.join(RESULTS_DIR, f"N{N}_T{T_STU}_L{L_TGT}_t{t}_g{g_int:03d}_{var}")


def _student_dense(t, g, var):
    prefix = _student_prefix(t, g, var)
    state = torch.load(prefix + "_state_dict.pt", map_location="cpu", weights_only=True)
    model = tdme.QMLM(N, T_STU)
    model.load_state_dict(state)
    model.eval()
    mpo = model.get_MPO(noise_type=NOISE)

    plain = []
    for tn_t in mpo.tensors:
        x = tn_t.unparametrize() if hasattr(tn_t, "unparametrize") else tn_t.copy()
        if torch.is_tensor(x.data):
            x.modify(data=x.data.to(torch.complex128))
        else:
            x.modify(data=np.asarray(x.data).astype(np.complex128))
        plain.append(x)
    tn = qtn.TensorNetwork(plain)
    M = tn.to_dense([f"k{i}" for i in range(N)], [f"input{i}" for i in range(N)])
    if torch.is_tensor(M):
        M = M.detach().cpu().numpy()
    M = np.asarray(M)
    assert np.max(np.abs(M.imag)) < 1e-9, "student PTM should be real"
    M_real = np.ascontiguousarray(M.real)
    del M, plain, tn, mpo, model, state
    gc.collect()
    return M_real


def _teacher_dense(t, g):
    gamma_vec = [g] * N
    M = build_tdme_dense(N, L_TGT, MU, gamma_vec, J=J_, t=t)
    return np.ascontiguousarray(M.astype(np.float64))


def _pair_distances(A, B, label=""):
    t0 = time.time()
    print(f"      [{label}] copy A ...  rss={_rss_gb():.1f}", flush=True)
    diff = A.copy()
    print(f"      [{label}] copy A done ({time.time()-t0:.0f}s)  rss={_rss_gb():.1f}", flush=True)
    t0 = time.time()
    diff -= B
    print(f"      [{label}] diff done ({time.time()-t0:.0f}s)  rss={_rss_gb():.1f}", flush=True)
    t0 = time.time()
    fro = float(np.linalg.norm(diff))
    print(f"      [{label}] fro done ({time.time()-t0:.0f}s, fro={fro:.4e})  rss={_rss_gb():.1f}",
          flush=True)
    t0 = time.time()
    print(f"      [{label}] starting gesvd...", flush=True)
    sv = sla.svd(diff, compute_uv=False, lapack_driver='gesvd',
                 overwrite_a=True, check_finite=False)
    tn = float(sv.sum())
    print(f"      [{label}] gesvd done ({time.time()-t0:.0f}s, trace={tn:.4e})  "
          f"rss={_rss_gb():.1f}", flush=True)
    del diff, sv
    gc.collect()
    return tn, fro


def _already_done(out_csv):
    done = set()
    if not os.path.exists(out_csv):
        return done
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            done.add((float(row["t"]), float(row["gamma"]), row["A"], row["B"]))
    return done


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pairs = list(itertools.combinations(CHANNELS, 2))  # C(6,2) = 15

    done = _already_done(OUT_CSV)
    write_header = not os.path.exists(OUT_CSV)
    fh = open(OUT_CSV, "a")
    if write_header:
        fh.write("t,gamma,A,B,trace_norm,fro_norm\n"); fh.flush()

    print(f"Grid: {len(T_LIST)} times x {len(GAMMA_LIST)} gammas = "
          f"{len(T_LIST)*len(GAMMA_LIST)} cells x {len(pairs)} pairs", flush=True)

    for t in T_LIST:
        for g in GAMMA_LIST:
            cell_done = all((t, g, a, b) in done for a, b in pairs)
            if cell_done:
                print(f"  SKIP t={t} g={g}  (all 15 pairs in CSV)", flush=True)
                continue
            tcell = time.time()
            print(f"  --- t={t}  g={g} ---  building 6 dense PTMs", flush=True)

            tts = time.time()
            tgt = _teacher_dense(t, g)
            print(f"    teacher built  ({time.time()-tts:.0f}s)  "
                  f"nbytes={tgt.nbytes/2**30:.1f}GB  rss={_rss_gb():.1f}GB", flush=True)

            mats = {"target": tgt}
            for var in VARIANTS:
                tts = time.time()
                mats[var] = _student_dense(t, g, var)
                gc.collect()
                print(f"    student {var:<10}  ({time.time()-tts:.0f}s)  rss={_rss_gb():.1f}GB",
                      flush=True)

            for (a, b) in pairs:
                if (t, g, a, b) in done:
                    continue
                tts = time.time()
                tn, fro = _pair_distances(mats[a], mats[b])
                gc.collect()
                fh.write(f"{t},{g},{a},{b},{tn:.6e},{fro:.6e}\n"); fh.flush()
                print(f"    pair ({a:<10}, {b:<10})  trace={tn:.4e}  fro={fro:.4e}  "
                      f"({time.time()-tts:.0f}s)  rss={_rss_gb():.1f}GB", flush=True)

            for k in list(mats.keys()):
                del mats[k]
            del mats
            gc.collect()
            print(f"  cell done in {(time.time()-tcell)/60:.1f} min", flush=True)

    fh.close()
    print(f"\nSaved {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
