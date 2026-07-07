"""
Parallel Trotterization TDSE
=============================
Parallelizes the (target_time × gamma_name) grid.
Each grid point calls Testing_TDME_Trotterization in a separate process.

Usage
-----
    python Trotterization_TDSE.py                # default workers
    python Trotterization_TDSE.py --workers 10   # 10 parallel workers
"""

import os, sys, signal, argparse, time
import numpy as np
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Where this script lives — spawned workers need this on sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Total CPU cores available (SLURM-aware)
# ---------------------------------------------------------------------------
TOTAL_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK",
                                  os.cpu_count() or 1))


# ===========================  worker function  =============================

def _worker(target_time, gamma_name, N, model_layer, model_to_learn_layer,
            mu, threads):
    """Run Testing_TDME_Trotterization for one (target_time, gamma_name) pair."""

    # -- pin thread counts BEFORE heavy imports --
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(threads)

    import torch
    torch.set_num_threads(threads)

    # make sure we can import project modules
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    import TDME_Trott as tdme                       # noqa: E402

    gamma = [gamma_name * 0.01] * N

    tag = f"time={target_time}, gamma_name={gamma_name}"
    print(f"[PID {os.getpid()}] START  {tag}  (threads={threads})",
          flush=True)

    testing_loss = tdme.Testing_TDME_Trotterization(
        N, model_layer,
        model_to_learn_layer=model_to_learn_layer,
        mu=mu, gamma=gamma, t=target_time,
    )

    print(f"[PID {os.getpid()}] DONE   {tag}  testing_loss={testing_loss}",
          flush=True)
    return target_time, gamma_name, float(testing_loss)


# ==============================  main  =====================================

def main():
    ap = argparse.ArgumentParser(
        description="Parallel Trotterization TDSE simulation")
    ap.add_argument("--workers", type=int, default=None,
                    help="number of parallel workers "
                         "(default: min(tasks, cores))")
    cli = ap.parse_args()

    # ---- simulation grid (matches the original script exactly) ------------
    N                    = 30
    model_layer          = 3
    model_to_learn_layer = 30
    mu                   = 1

    target_times = [0.2, 0.4, 0.6, 0.8]
    gamma_names  = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]

    # ---- build flat list of grid points -----------------------------------
    grid = [(t, g) for t in target_times for g in gamma_names]
    n_tasks = len(grid)

    # ---- resource allocation ----------------------------------------------
    n_workers = cli.workers or min(n_tasks, TOTAL_CORES)
    n_workers = min(n_workers, n_tasks, TOTAL_CORES)
    threads   = max(1, TOTAL_CORES // n_workers)

    print("=" * 60)
    print(f"  CPU cores      : {TOTAL_CORES}")
    print(f"  Grid points    : {n_tasks}  "
          f"({len(target_times)} times × {len(gamma_names)} gammas)")
    print(f"  Workers        : {n_workers}")
    print(f"  Threads/worker : {threads}")
    print("=" * 60, flush=True)

    # ---- launch pool (spawn avoids inheriting parent thread-pools) --------
    ctx  = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)

    futures = {}
    for t_val, g_name in grid:
        fut = pool.submit(
            _worker,
            t_val, g_name,
            N, model_layer, model_to_learn_layer,
            mu, threads,
        )
        futures[fut] = (t_val, g_name)

    # ---- graceful Ctrl-C --------------------------------------------------
    def _abort(sig=None, frame=None):
        print("\n⚠  Interrupted — killing workers …", flush=True)
        for f in futures:
            f.cancel()
        for pid in list(pool._processes):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        os._exit(1)

    signal.signal(signal.SIGINT, _abort)

    # ---- collect results --------------------------------------------------
    # Store results grouped by target_time
    results = {t: {} for t in target_times}

    t0 = time.time()
    done, failed = 0, 0
    try:
        for fut in as_completed(futures):
            t_val, g_name = futures[fut]
            try:
                _, _, loss = fut.result()
                results[t_val][g_name] = loss
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  time={t_val}, "
                      f"gamma_name={g_name}  loss={loss:.6f}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  time={t_val}, gamma_name={g_name}: {exc}",
                      flush=True)
    except KeyboardInterrupt:
        _abort()
    finally:
        pool.shutdown(wait=True)

    # ---- save results (one .npy per target_time) --------------------------
    for t_val in target_times:
        loss_list = [results[t_val].get(g, float("nan"))
                     for g in gamma_names]
        fname = f"Trotterization_using_3_layer_to_30_layer_time_{t_val}.npy"
        np.save(fname, np.array(loss_list))
        print(f"  Saved {fname}  →  {loss_list}")

    elapsed = time.time() - t0
    print(f"\nFinished {done} tasks ({failed} failed) "
          f"in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()