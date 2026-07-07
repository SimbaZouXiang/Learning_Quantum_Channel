"""For each (L, p) cell, compute all C(5,2)=10 inter-student pairwise distances
on the three eval sets (weight1full=24, weight2full=252, random500=500).

Reuses the saved student parameters from mpo_save_params and mpo_random_test.
Per input: build each student's output once, then compute all 10 pairwise
tensor_network_distance values + 5 student-vs-target distances. Average per
(eval_set, pair) and write to a single CSV.
"""
import os, sys, signal, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
import itertools

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]
MAX_BD, MAX_ERR = 32, 1e-6
NOISE_TYPE = "depolarizing"
TRUNCATION = False
N_RANDOM = 500
RANDOM_SEED = 12345

VARIANTS = [
    ("w1",        "mpo_save_params"),
    ("w2full",    "mpo_save_params"),
    ("combined",  "mpo_save_params"),
    ("random24",  "mpo_random_test"),
    ("random276", "mpo_random_test"),
]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _load_student(L, p, variant_tag, folder):
    import torch
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme
    prefix = os.path.join(PARENT_DIR, folder, "results",
                           f"N{N}_T{T}_L{L}_p{p}_{variant_tag}")
    params_np    = np.load(prefix + "_params.npy",   allow_pickle=True)
    p_depolar    = np.load(prefix + "_p_depolar.npy", allow_pickle=True)
    p_dephaseX   = np.load(prefix + "_p_dephaseX.npy", allow_pickle=True)
    p_dephaseY   = np.load(prefix + "_p_dephaseY.npy", allow_pickle=True)
    p_dephaseZ   = np.load(prefix + "_p_dephaseZ.npy", allow_pickle=True)
    model = tdme.QMLM(N, T, max_bd=MAX_BD, max_err=MAX_ERR)
    model.params = torch.nn.Parameter(
        torch.from_numpy(params_np).to(torch.complex128), requires_grad=False,
    )
    model.p_depolar.data    = torch.from_numpy(p_depolar).to(torch.float64)
    model.p_dephaseX.data   = torch.from_numpy(p_dephaseX).to(torch.float64)
    model.p_dephaseY.data   = torch.from_numpy(p_dephaseY).to(torch.float64)
    model.p_dephaseZ.data   = torch.from_numpy(p_dephaseZ).to(torch.float64)
    return model


def _build_target_module(L, p):
    import torch
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme
    target_param = torch.load(
        os.path.join(PARENT_DIR, "mpo_save_params", "results",
                      f"N{N}_T{T}_L{L}_p{p}_w1_target_param.pt"),
        map_location="cpu", weights_only=True,
    )
    p_depolar_tgt = torch.ones((L, N), dtype=torch.float64) * p
    return tdme.QMLM_output_only(
        N, L, param=target_param, p_depolar=p_depolar_tgt,
        max_bd=MAX_BD, max_err=MAX_ERR,
    )


def _worker(L_target, depol):
    threads = 18
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(threads)
    import torch
    torch.set_num_threads(threads)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme

    print(f"[PID {os.getpid()}] START  L={L_target} p={depol}", flush=True)
    t0 = time.time()
    students = {tag: _load_student(L_target, depol, tag, folder)
                for tag, folder in VARIANTS}
    student_tags = [tag for tag, _ in VARIANTS]
    target_module = _build_target_module(L_target, depol)
    student_mpos = {tag: m.get_MPO(noise_type=NOISE_TYPE) for tag, m in students.items()}

    inputs_w1 = tdme.Pauli_MPS_weight_1(N)
    inputs_w2 = tdme.Pauli_MPS_weight_2_full(N)
    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    inputs_rand = [tdme.random_pauli_MPS(N)[0] for _ in range(N_RANDOM)]
    eval_sets = [("weight1full", inputs_w1),
                  ("weight2full", inputs_w2),
                  ("random500",   inputs_rand)]

    pair_keys = list(itertools.combinations(student_tags, 2))   # 10 pairs
    tgt_keys = student_tags                                      # 5 vs target
    sums = {(es, "tgt-"+t): 0.0 for es, _ in eval_sets for t in tgt_keys}
    sums.update({(es, f"{a}-{b}"): 0.0 for es, _ in eval_sets for (a, b) in pair_keys})
    counts = {es: 0 for es, _ in eval_sets}

    with torch.no_grad():
        for es_name, inputs in eval_sets:
            for inp in inputs:
                tgt = target_module.forward_depolarizing_only(
                    inp.copy(), truncation=TRUNCATION,
                )
                for i, ten in enumerate(tgt):
                    ten.reindex_({f'input{i}': f'k{i}'})
                tgt_c = tgt.astype("complex128")
                outs = {tag: (student_mpos[tag] | inp).astype("complex128")
                         for tag in student_tags}
                for tag in student_tags:
                    sums[(es_name, "tgt-"+tag)] += tdme.tensor_network_distance(
                        outs[tag], tgt_c
                    ).item()
                for (a, b) in pair_keys:
                    sums[(es_name, f"{a}-{b}")] += tdme.tensor_network_distance(
                        outs[a], outs[b]
                    ).item()
                counts[es_name] += 1
    elapsed = time.time() - t0
    print(f"[PID {os.getpid()}] DONE   L={L_target} p={depol}  ({elapsed:.0f}s)", flush=True)
    return L_target, depol, sums, counts


def main():
    grid = [(L, p) for L in LTARGETS for p in DEPOL_LIST]
    n_workers = min(len(grid), 10)
    print(f"  Tasks: {len(grid)}  Workers: {n_workers}", flush=True)

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {pool.submit(_worker, L, p): (L, p) for L, p in grid}

    def _abort(sig=None, frame=None):
        for f in futures: f.cancel()
        for pid in list(pool._processes):
            try: os.kill(pid, signal.SIGKILL)
            except OSError: pass
        os._exit(1)
    signal.signal(signal.SIGINT, _abort)

    rows = {}
    t0 = time.time()
    try:
        for fut in as_completed(futures):
            L, p = futures[fut]
            try:
                _L, _p, sums, counts = fut.result()
                rows[(L, p)] = (sums, counts)
                print(f"  ✓  L={L} p={p}", flush=True)
            except Exception as exc:
                print(f"  ✗  L={L} p={p}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)
    print(f"Finished in {(time.time()-t0)/60:.1f} min", flush=True)

    student_tags = [tag for tag, _ in VARIANTS]
    pair_keys = list(itertools.combinations(student_tags, 2))
    eval_names = ["weight1full", "weight2full", "random500"]
    cols = ["L", "p", "eval_set"]
    cols += [f"tgt-{t}" for t in student_tags]
    cols += [f"{a}-{b}" for (a, b) in pair_keys]
    csv = os.path.join(RESULTS_DIR, "pairwise_5variant.csv")
    with open(csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for L in LTARGETS:
            for p in DEPOL_LIST:
                if (L, p) not in rows: continue
                sums, counts = rows[(L, p)]
                for es in eval_names:
                    nc = counts[es]
                    line = [str(L), str(p), es]
                    for t in student_tags:
                        line.append(f"{sums[(es,'tgt-'+t)]/nc:.6e}")
                    for (a, b) in pair_keys:
                        line.append(f"{sums[(es,f'{a}-{b}')]/nc:.6e}")
                    f.write(",".join(line) + "\n")
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
