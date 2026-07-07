"""Extended evaluation of trained student MPOs on larger input sets.

For each (L, p) cell we have 5 trained students saved:
  w1         (24 structured w1 training inputs)            — mpo_save_params
  w2full     (252 structured w2-full training inputs)      — mpo_save_params
  combined   (276 structured w1+w2-full training inputs)   — mpo_save_params
  random24   (24 random Pauli training inputs)             — mpo_random_test
  random276  (276 random Pauli training inputs)            — mpo_random_test

Evaluate each on three input sets:
  weight1full   (24 inputs, complete weight-1 basis)
  weight2full   (252 inputs, complete weight-2 basis — full operator pairs)
  random500     (500 random Pauli MPS, fixed-seed)

Per-cell strategy: compute target output ONCE per input, reuse against all 5
student MPOs. 10 cells × parallel workers makes this comfortable in 1 h.
"""
import os, sys, signal, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

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
    threads = 18  # 10 cells × 18 threads ≈ 180 cores
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
    # Load all 5 students.
    students = {
        tag: _load_student(L_target, depol, tag, folder)
        for tag, folder in VARIANTS
    }
    # Target module.
    target_module = _build_target_module(L_target, depol)

    # Pre-compute eval input sets.
    inputs_w1full = tdme.Pauli_MPS_weight_1(N)
    inputs_w2full = tdme.Pauli_MPS_weight_2_full(N)
    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    inputs_random = [tdme.random_pauli_MPS(N)[0] for _ in range(N_RANDOM)]
    eval_sets = [
        ("weight1full", inputs_w1full),
        ("weight2full", inputs_w2full),
        ("random500",   inputs_random),
    ]

    # For each eval set, loop over inputs; for each input, compute target output
    # ONCE, then evaluate against each student MPO.
    student_mpos = {tag: m.get_MPO(noise_type=NOISE_TYPE) for tag, m in students.items()}
    per_variant_losses = {(tag, es): [] for tag, _ in VARIANTS for es, _ in eval_sets}

    with torch.no_grad():
        for es_name, inputs in eval_sets:
            for inp in inputs:
                tgt = target_module.forward_depolarizing_only(
                    inp.copy(), truncation=TRUNCATION,
                )
                for i, ten in enumerate(tgt):
                    ten.reindex_({f'input{i}': f'k{i}'})
                tgt_c = tgt.astype("complex128")
                for tag, _ in VARIANTS:
                    student_out = (student_mpos[tag] | inp).astype("complex128")
                    loss = tdme.tensor_network_distance(student_out, tgt_c).item()
                    per_variant_losses[(tag, es_name)].append(loss)

    # Save per-variant per-evalset arrays.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}
    for tag, _ in VARIANTS:
        for es_name, _ in eval_sets:
            arr = np.asarray(per_variant_losses[(tag, es_name)], dtype=float)
            fn = f"N{N}_T{T}_L{L_target}_p{depol}_{tag}_eval_{es_name}.npy"
            np.save(os.path.join(RESULTS_DIR, fn), arr)
            summary[(tag, es_name)] = arr.mean()

    elapsed = time.time() - t0
    print(f"[PID {os.getpid()}] DONE   L={L_target} p={depol}  ({elapsed:.0f}s)",
          flush=True)
    for tag, _ in VARIANTS:
        means = [summary[(tag, es)] for es, _ in eval_sets]
        print(f"  {tag:>10s}  w1full={means[0]:.4e}  w2full={means[1]:.4e}  "
              f"rand500={means[2]:.4e}", flush=True)
    return L_target, depol


def main():
    grid = [(L, p) for L in LTARGETS for p in DEPOL_LIST]
    n_tasks = len(grid)
    n_workers = min(n_tasks, 10)
    print(f"  Tasks: {n_tasks}  Workers: {n_workers}", flush=True)
    print(f"  L_targets={LTARGETS}  depols={DEPOL_LIST}", flush=True)
    print(f"  Variants: {[v[0] for v in VARIANTS]}", flush=True)
    print(f"  Eval sets: w1full=24, w2full=252, random={N_RANDOM}", flush=True)

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

    t0 = time.time()
    done, failed = 0, 0
    try:
        for fut in as_completed(futures):
            L, p = futures[fut]
            try:
                _ = fut.result()
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  L={L} p={p}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  L={L} p={p}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)
    print(f"\nFinished {done} ({failed} failed) in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
