"""Production driver: train all 200 (t, gamma, variant) cells.

Grid: t in {0.5, 1.0, 2.0, 3.0} x gamma in {0, 0.02, ..., 0.5} x 5 variants.
Each cell:
  - Learning_TDME_scheduler (200 epochs + fine-tune)
  - Save state_dict + params + noise rates + losses + meta.json
  - Skip if meta.json already present (resume)

Parallel: ProcessPoolExecutor across cells; threads_per_worker set by env.
Configure THREADS_PER_WORKER from the smoke run's measured per-cell cost.
"""
import os, sys, time, json, signal
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- experiment grid ----------
N         = 8
T_STUDENT = 3
L_TARGET  = 10
MU        = 1
J         = 1
EPOCHS    = 200
LR        = 0.01
MAX_BD    = 64
MAX_ERR   = 1e-6
TRUNCATION = False
USE_COMPRESSED = False
USE_SCHED  = True
NOISE_TYPE = "dephasing"
SEED       = 12345

T_LIST     = [0.5, 1.0, 2.0, 3.0]
GAMMA_LIST = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50]

# All 5 variants. Filter at run-time via TDME_VARIANTS env var.
ALL_VARIANTS = [
    ("w1",        1),
    ("w2full",    "2_full"),
    ("combined",  "combined"),
    ("random24",  ("random", 24,  SEED)),
    ("random276", ("random", 276, SEED)),
]
_filter = os.environ.get("TDME_VARIANTS", "").split(",") if os.environ.get("TDME_VARIANTS") else None
VARIANTS = [(t, a) for t, a in ALL_VARIANTS if _filter is None or t in _filter]

# Parallelism: set by env from job script. Smoke run will tell us a sensible default.
THREADS_PER_WORKER = int(os.environ.get("TDME_THREADS_PER_WORKER", 24))
TOTAL_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

def _cell_prefix(t, gamma, variant_tag):
    g_int = int(round(gamma * 100))
    return os.path.join(RESULTS_DIR,
        f"N{N}_T{T_STUDENT}_L{L_TARGET}_t{t}_g{g_int:03d}_{variant_tag}")


def _is_done(t, gamma, variant_tag):
    return os.path.exists(_cell_prefix(t, gamma, variant_tag) + "_meta.json")


def _worker(t, gamma, variant_tag, input_pw, threads):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(threads)
    os.environ.setdefault("COTENGRA_PARALLEL", "false")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    import torch
    torch.set_num_threads(threads)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme

    prefix = _cell_prefix(t, gamma, variant_tag)
    fn_tag = os.path.basename(prefix)
    if os.path.exists(prefix + "_meta.json"):
        print(f"[PID {os.getpid()}] SKIP   {fn_tag}  (meta.json exists)", flush=True)
        return fn_tag, "skipped", 0.0

    # per-cell deterministic seed: same input set across cells for given variant
    np.random.seed(SEED); torch.manual_seed(SEED)
    # steer in-function checkpoint into a per-cell file so concurrent workers
    # don't clobber each other
    os.environ["LEARNING_TDME_CKPT_DIR"] = RESULTS_DIR
    gamma_vec = [gamma] * N

    print(f"[PID {os.getpid()}] START  {fn_tag}  threads={threads}", flush=True)
    t0 = time.time()
    try:
        model, learning_loss, testing_loss, testing_loss_list = \
            tdme.Learning_TDME_scheduler(
                N=N, MPO_layer=T_STUDENT, model_to_learn_layer=L_TARGET,
                mu=MU, gamma=gamma_vec, J=J, t=t,
                epochs=EPOCHS, lr=LR, normalized=False,
                max_bd=MAX_BD, max_err=MAX_ERR,
                truncation=TRUNCATION,
                noise_type=NOISE_TYPE,
                use_scheduler=USE_SCHED,
                use_compressed=USE_COMPRESSED,
                num_threads=1,  # sequential data gen + testing inside the worker; outer pool handles cells in parallel
                data_dir=None,
                input_pauli_weight=input_pw,
            )
    except Exception as e:
        print(f"[PID {os.getpid()}] FAIL   {fn_tag}: {type(e).__name__}: {e}", flush=True)
        raise
    elapsed = time.time() - t0

    torch.save(model.state_dict(), prefix + "_state_dict.pt")
    np.save(prefix + "_params.npy",     model.params.detach().cpu().numpy())
    np.save(prefix + "_p_depolar.npy",  model.p_depolar.detach().cpu().numpy())
    np.save(prefix + "_p_dephaseX.npy", model.p_dephaseX.detach().cpu().numpy())
    np.save(prefix + "_p_dephaseY.npy", model.p_dephaseY.detach().cpu().numpy())
    np.save(prefix + "_p_dephaseZ.npy", model.p_dephaseZ.detach().cpu().numpy())
    np.save(prefix + "_learning_loss.npy",     np.array(learning_loss))
    np.save(prefix + "_testing_loss.npy",      np.array(testing_loss))
    np.save(prefix + "_testing_loss_list.npy", np.array(testing_loss_list))

    meta = dict(
        N=N, T_student=T_STUDENT, L_target=L_TARGET, mu=MU,
        gamma=float(gamma), gamma_vec=list(gamma_vec),
        J=J, t=float(t),
        variant=variant_tag,
        input_pauli_weight=(input_pw if not isinstance(input_pw, tuple)
                             else list(input_pw)),
        lr=LR, epochs=EPOCHS,
        max_bd=MAX_BD, max_err=MAX_ERR,
        truncation=TRUNCATION,
        use_compressed=USE_COMPRESSED, use_scheduler=USE_SCHED,
        noise_type=NOISE_TYPE,
        random_seed=SEED,
        elapsed_seconds=elapsed,
        final_testing_loss=float(testing_loss),
    )
    with open(prefix + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[PID {os.getpid()}] DONE   {fn_tag}  ({elapsed:.0f}s)", flush=True)
    return fn_tag, "done", elapsed


def main():
    grid = [(t, g, vt, va) for t in T_LIST for g in GAMMA_LIST for vt, va in VARIANTS]
    remaining = [c for c in grid if not _is_done(c[0], c[1], c[2])]
    print(f"  Grid: {len(grid)}  Remaining: {len(remaining)}", flush=True)
    if not remaining:
        print("All cells complete.", flush=True)
        return

    n_workers = max(1, TOTAL_CORES // THREADS_PER_WORKER)
    n_workers = min(n_workers, len(remaining))
    print(f"  Cores={TOTAL_CORES}  Workers={n_workers}  Threads/worker={THREADS_PER_WORKER}",
          flush=True)

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {pool.submit(_worker, t, g, vt, va, THREADS_PER_WORKER): (t, g, vt)
               for (t, g, vt, va) in remaining}

    def _abort(sig=None, frame=None):
        for f in futures: f.cancel()
        for pid in list(pool._processes):
            try: os.kill(pid, signal.SIGKILL)
            except OSError: pass
        os._exit(1)
    signal.signal(signal.SIGINT, _abort)

    t0 = time.time()
    n_done = 0
    try:
        for fut in as_completed(futures):
            t, g, vt = futures[fut]
            try:
                fn, status, elapsed = fut.result()
                n_done += 1
                print(f"  [{n_done}/{len(remaining)}] {status} {fn} ({elapsed:.0f}s)", flush=True)
            except Exception as exc:
                print(f"  FAIL {vt} t={t} g={g}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)
    print(f"\n  Total wall: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
