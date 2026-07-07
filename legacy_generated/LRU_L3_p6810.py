"""
Parallel main.py — runs Learning_MPO_scheduler for all (depolarizing_name, L)
combinations using Python multiprocessing.

Usage:
    python main.py          # uses all available CPUs (or NUMBA_NUM_THREADS)
    python main.py --workers 6   # explicit number of workers

With 192 CPUs and 6 workers, each worker gets ~32 threads for
any internal numba / BLAS parallelism.
"""

import os
import sys
import signal
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# ── Determine available cores ──────────────────────────────────────────
TOTAL_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK",
                  os.cpu_count() or 1))


def run_single_task(args):
    """Run one (depolarizing_name, L) combination in its own process.

    Each worker limits its own thread usage so that the total across all
    workers does not exceed the available cores.
    """
    depolarizing_name, N, T, L, epochs, lr, normalized, truncation, noise_type, threads_per_worker = args

    # ── Limit threads BEFORE importing heavy libraries ──
    # For small matrices (N=8, max_bd=32), multi-threaded BLAS causes massive
    # slowdowns due to thread contention. We lock BLAS to 1 thread, and instead
    # use Python-level ThreadPoolExecutor across independent samples by passing
    # `num_threads` into Learning_MPO_scheduler.
    os.environ["NUMBA_NUM_THREADS"]      = "1"
    os.environ["OMP_NUM_THREADS"]        = "1"
    os.environ["MKL_NUM_THREADS"]        = "1"
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"

    import torch
    torch.set_num_threads(1)

    import TDME_Trott as tdme

    depolarizing_strength = depolarizing_name * 0.01
    print(f"[Worker PID {os.getpid()}] depol={depolarizing_strength}, "
          f"L={L}, threads={threads_per_worker}", flush=True)

    # Return order from Learning_MPO_scheduler:
    # model, learning_loss, param, p_depolar, testing_loss, testing_loss_list,
    # model_params, model_p_depolar, model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ
    (model, learning_loss, target_param, p_depolar,
     testing_loss, testing_loss_list, model_param, model_p_depolar,
     model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ
    ) = tdme.Learning_MPO_scheduler(
        N, T, L,
        depolarizing_strength=depolarizing_strength,
        epochs=epochs, lr=lr,
        normalized=normalized,
        truncation=truncation,
        noise_type=noise_type,
        num_threads=threads_per_worker,  # Use ThreadPool inside
        use_compressed=False,
        max_bd=32,
    )

    # ── Save results (each worker writes its own files) ────────────────
    prefix = f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}"
    np.save(f"{prefix}_learning_loss.npy",     np.array(learning_loss))
    np.save(f"{prefix}_testing_loss.npy",      np.array(testing_loss))
    np.save(f"{prefix}_testing_loss_list.npy", np.array(testing_loss_list))
    np.save(f"{prefix}_target_param.npy",      np.array(target_param))
    np.save(f"{prefix}_model_param.npy",       np.array(model_param))
    np.save(f"{prefix}_model_p_depolar.npy",   np.array(model_p_depolar))

    print(f"[Worker PID {os.getpid()}] DONE — depol={depolarizing_strength}, L={L}, "
          f"testing_loss={testing_loss}", flush=True)

    return depolarizing_name, L, float(testing_loss)


def main():
    parser = argparse.ArgumentParser(description="Parallel Learning_MPO_scheduler")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: one per task)")
    args = parser.parse_args()

    # ── Problem parameters ─────────────────────────────────────────────
    N = 30
    T = 3
    depolarizing_names = [6, 8, 10]
    target_layers      = [3]
    epochs       = 150
    lr           = 0.05
    normalized   = False
    truncation   = True
    noise_type   = "depolarizing"

    # ── Build task list ────────────────────────────────────────────────
    tasks = []
    for depolarizing_name in depolarizing_names:
        for L in target_layers:
            tasks.append((depolarizing_name, N, T, L, epochs, lr,
                          normalized, truncation, noise_type))

    num_tasks   = len(tasks)
    num_workers = args.workers if args.workers else min(num_tasks, 6)
    num_workers = min(num_workers, num_tasks, TOTAL_CORES)

    # Give each worker a fair share of cores (for BLAS/OMP, not Python threads)
    # Cap thread count to prevent OOM when computing tensor networks for large L (L=6,7)
    threads_per_worker = min(8, max(1, TOTAL_CORES // num_workers))

    # Append threads_per_worker to each task tuple
    tasks = [t + (threads_per_worker,) for t in tasks]

    print("=" * 60)
    print(f"Total cores available : {TOTAL_CORES}")
    print(f"Number of tasks       : {num_tasks}")
    print(f"Number of workers     : {num_workers}")
    print(f"Threads per worker    : {threads_per_worker}")
    print("=" * 60, flush=True)

    t0 = time.time()

    # ── Launch workers ─────────────────────────────────────────────────
    # Use 'spawn' to ensure each child process gets clean env variables
    # (fork can inherit parent's thread-pool, causing oversubscription).
    from multiprocessing import get_context
    ctx = get_context("spawn")

    pool = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
    futures = {pool.submit(run_single_task, task): task for task in tasks}

    # ── Install signal handler so Ctrl+C kills everything ──────────────
    def _kill_all_workers(signum=None, frame=None):
        print("\n⚠ Ctrl+C received — force-killing all workers...", flush=True)
        # Cancel pending futures
        for f in futures:
            f.cancel()
        # Force-kill every worker process via SIGKILL
        for pid in list(pool._processes.keys()):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        # Use os._exit to skip atexit handlers (they hang on join())
        os._exit(1)

    signal.signal(signal.SIGINT, _kill_all_workers)

    # ── Collect results ────────────────────────────────────────────────
    try:
        for future in as_completed(futures):
            task = futures[future]
            depolarizing_name, _, _, L = task[0], task[1], task[2], task[3]
            try:
                dep_name, layer, test_loss = future.result()
                print(f"✓ Completed: depol_name={dep_name}, L={layer}, "
                      f"testing_loss={test_loss:.6f}", flush=True)
            except Exception as exc:
                print(f"✗ FAILED: depol_name={depolarizing_name}, L={L}: {exc}",
                      flush=True)
    except KeyboardInterrupt:
        _kill_all_workers()
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"\nAll tasks finished in {elapsed:.1f} s  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
