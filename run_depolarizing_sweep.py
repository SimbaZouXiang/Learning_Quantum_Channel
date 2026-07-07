"""Parameterized depolarizing-noise learning sweep.

Replaces the 14 code-generated LRU_p*_L*.py clones (now archived in
legacy_generated/): one worker process per (p, L) grid point, each calling
qcl's Learning_MPO_scheduler and saving the same six .npy outputs with the
same Depolarizing_N{N}_T{T}_L{L}_p_{p:03d}_* naming as before.

Examples:
    python run_depolarizing_sweep.py --p-list 0 2 4 --L-list 3 4 5
    python run_depolarizing_sweep.py --p-list 6 8 10 --L-list 6 7 \
        --threads-per-worker 8 --outdir Result_random_unitary
"""

import os
import sys
import signal
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTAL_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def run_single_task(args):
    """Run one (depolarizing_name, L) combination in its own process."""
    (depolarizing_name, N, T, L, epochs, lr, normalized, truncation,
     noise_type, use_compressed, max_bd, threads_per_worker, outdir) = args

    # Limit threads BEFORE importing heavy libraries: multi-threaded BLAS on
    # small matrices causes contention; per-sample parallelism happens at the
    # Python level via num_threads instead.
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import torch
    torch.set_num_threads(1)

    sys.path.insert(0, SCRIPT_DIR)
    from qcl import Learning_MPO_scheduler

    depolarizing_strength = depolarizing_name * 0.01
    print(f"[Worker PID {os.getpid()}] depol={depolarizing_strength}, "
          f"L={L}, threads={threads_per_worker}", flush=True)

    (model, learning_loss, target_param, p_depolar,
     testing_loss, testing_loss_list, model_param, model_p_depolar,
     model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ
     ) = Learning_MPO_scheduler(
        N, T, L,
        depolarizing_strength=depolarizing_strength,
        epochs=epochs, lr=lr,
        normalized=normalized,
        truncation=truncation,
        noise_type=noise_type,
        num_threads=threads_per_worker,
        use_compressed=use_compressed,
        max_bd=max_bd,
    )

    prefix = os.path.join(outdir, f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}")
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p-list", type=int, nargs="+", required=True,
                        help="depolarizing strengths in percent, e.g. 0 2 4 (p=0.00, 0.02, 0.04)")
    parser.add_argument("--L-list", type=int, nargs="+", required=True,
                        help="target (teacher) layer counts, e.g. 3 4 5")
    parser.add_argument("--N", type=int, default=30, help="number of qubits (default 30)")
    parser.add_argument("--T", type=int, default=3, help="student MPO layers (default 3)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-bd", type=int, default=32)
    parser.add_argument("--use-compressed", action="store_true",
                        help="use the layer-by-layer compressed forward path")
    parser.add_argument("--no-truncation", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker processes (default: min(#tasks, 6))")
    parser.add_argument("--threads-per-worker", type=int, default=None,
                        help="per-worker sample threads (default 16, use 8 for L>=6 to avoid OOM)")
    parser.add_argument("--outdir", type=str, default=".",
                        help="directory for the .npy outputs (default: CWD, matching the old scripts)")
    args = parser.parse_args()

    normalized = False
    truncation = not args.no_truncation
    noise_type = "depolarizing"

    os.makedirs(args.outdir, exist_ok=True)

    tasks = []
    for p in args.p_list:
        for L in args.L_list:
            tasks.append((p, args.N, args.T, L, args.epochs, args.lr,
                          normalized, truncation, noise_type,
                          args.use_compressed, args.max_bd))

    num_tasks = len(tasks)
    num_workers = args.workers if args.workers else min(num_tasks, 6)
    num_workers = min(num_workers, num_tasks, TOTAL_CORES)
    threads_per_worker = args.threads_per_worker or min(16, max(1, TOTAL_CORES // num_workers))
    tasks = [t + (threads_per_worker, args.outdir) for t in tasks]

    print("=" * 60)
    print(f"Total cores available : {TOTAL_CORES}")
    print(f"Number of tasks       : {num_tasks}")
    print(f"Number of workers     : {num_workers}")
    print(f"Threads per worker    : {threads_per_worker}")
    print(f"Output directory      : {args.outdir}")
    print("=" * 60, flush=True)

    t0 = time.time()

    # 'spawn' so children get clean env (fork inherits parent thread pools).
    from multiprocessing import get_context
    ctx = get_context("spawn")

    pool = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
    futures = {pool.submit(run_single_task, task): task for task in tasks}

    def _kill_all_workers(signum=None, frame=None):
        print("\nCtrl+C received — force-killing all workers...", flush=True)
        for f in futures:
            f.cancel()
        for pid in list(pool._processes.keys()):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        os._exit(1)

    signal.signal(signal.SIGINT, _kill_all_workers)

    try:
        for future in as_completed(futures):
            task = futures[future]
            p, L = task[0], task[3]
            try:
                dep_name, layer, test_loss = future.result()
                print(f"Completed: p={dep_name}, L={layer}, "
                      f"testing_loss={test_loss:.6f}", flush=True)
            except Exception as exc:
                print(f"FAILED: p={p}, L={L}: {exc}", flush=True)
    except KeyboardInterrupt:
        _kill_all_workers()
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"\nAll tasks finished in {elapsed:.1f} s  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
