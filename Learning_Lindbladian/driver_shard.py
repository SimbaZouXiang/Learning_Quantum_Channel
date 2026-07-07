"""Generic single-node shard driver for the (target_time, gamma) grid.

Same code path as Learning_TDME_using_data_*.py, but accepts `--target_time`
and `--gammas` (comma-separated) on the command line so we can shard the
10-gamma grid across multiple SLURM jobs / nodes.

Per-node default: 5 workers x 32 threads (= 160 of 192 cores active).
That's the contention sweet spot we identified: above 5 concurrent
workers, per-worker throughput collapses; at 5 it stays close to the
3-worker calibration (~8-12 h per grid point).
"""

import os, sys, signal, argparse, time
import numpy as np
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from joblib.externals.loky import get_reusable_executor as _loky_executor
    _HAS_LOKY = True
except ImportError:
    _HAS_LOKY = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTAL_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def _worker(target_time, gamma_name, N, T, model_to_learn_layer,
            mu, J, epochs, lr, normalized, truncation, noise_type,
            use_scheduler, use_compressed, max_bd, threads):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ.setdefault("COTENGRA_PARALLEL", "false")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    import torch
    torch.set_num_threads(threads)

    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    for p in (PARENT_DIR, SCRIPT_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    import TDME_Trott as tdme

    gamma = [gamma_name * 0.01] * N
    data_dir = os.path.join(
        SCRIPT_DIR, "Learning_data",
        f"N{N}_T{model_to_learn_layer}_mu{float(mu)}_gamma{round(gamma[0], 2)}_t{target_time}",
    )
    tag = f"time={target_time}, gamma_name={gamma_name}"
    print(f"[PID {os.getpid()}] START {tag}  (threads={threads})", flush=True)

    t0 = time.time()
    model, learning_loss, testing_loss, testing_loss_list = \
        tdme.Learning_TDME_scheduler(
            N, T,
            model_to_learn_layer=model_to_learn_layer,
            mu=mu, gamma=gamma, J=J, t=target_time,
            epochs=epochs, lr=lr,
            normalized=normalized,
            truncation=truncation,
            noise_type=noise_type,
            use_scheduler=use_scheduler,
            use_compressed=use_compressed,
            max_bd=max_bd,
            data_dir=data_dir,
        )
    wall = time.time() - t0

    prefix = (f"Learning_result/TDME_N{N}_T{T}_Modeltolearnlayer{model_to_learn_layer}"
              f"_time{target_time}_gamma{gamma_name:03d}")
    np.save(f"{prefix}_learning_loss.npy",     np.array(learning_loss))
    np.save(f"{prefix}_testing_loss.npy",      np.array(testing_loss))
    np.save(f"{prefix}_testing_loss_list.npy", np.array(testing_loss_list))

    print(f"[PID {os.getpid()}] DONE  {tag}  testing_loss={testing_loss}  wall={wall:.0f}s",
          flush=True)
    return target_time, gamma_name, float(testing_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_time", type=float, required=True)
    ap.add_argument("--gammas", type=str, required=True,
                    help="comma-separated integer gamma_names, e.g. 0,2,4,6,8")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: min(n_tasks, TOTAL_CORES // 32)")
    ap.add_argument("--threads", type=int, default=16,
                    help="BLAS threads per worker (default: 16, matching the "
                         "old-run benchmark sweet spot)")
    cli = ap.parse_args()

    target_time  = cli.target_time
    gamma_names  = [int(g) for g in cli.gammas.split(",") if g.strip()]

    # ---- model / training params (matches Learning_TDME_using_data_*.py) ---
    N                    = 30
    T                    = 3
    model_to_learn_layer = 30
    mu                   = 1
    J                    = 1
    epochs               = 200
    lr                   = 0.05
    normalized           = False
    truncation           = True
    noise_type           = "dephasing"
    use_scheduler        = False
    use_compressed       = False
    max_bd               = 64

    grid    = [(target_time, g) for g in gamma_names]
    n_tasks = len(grid)

    # 32 threads/worker is the sweet spot for this contention profile (BLAS
    # scaling already plateaus past ~16 threads, so adding more threads costs
    # nothing useful, but spawning more *workers* hits the bandwidth wall
    # we saw at 10x19 = 190 threads).
    # 16 threads/worker is the BLAS plateau on this code path. Past ~16
    # additional threads do nothing useful per worker, but TOTAL active BLAS
    # threads across all workers drives memory-bandwidth contention. Old
    # 3x16=48-thread runs finished a grid point in ~8 h; 5x38=190-thread
    # and 10x19=190-thread runs both took 24 h+. So we cap threads here
    # explicitly rather than letting it grow with n_workers.
    threads = cli.threads
    n_workers = (cli.workers
                 or min(n_tasks, max(1, TOTAL_CORES // max(threads, 1))))
    n_workers = min(n_workers, n_tasks, TOTAL_CORES)

    print("=" * 60)
    print(f"  target_time    : {target_time}")
    print(f"  gamma_names    : {gamma_names}")
    print(f"  CPU cores      : {TOTAL_CORES}")
    print(f"  Grid points    : {n_tasks}")
    print(f"  Workers        : {n_workers}")
    print(f"  Threads/worker : {threads}")
    print("=" * 60, flush=True)

    if _HAS_LOKY:
        print("Using loky reusable executor (worker-crash-resilient).", flush=True)
        pool = _loky_executor(max_workers=n_workers, context="spawn", timeout=None)
    else:
        ctx = get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)

    futures = {}
    for t_val, g_name in grid:
        fut = pool.submit(
            _worker, t_val, g_name,
            N, T, model_to_learn_layer,
            mu, J, epochs, lr, normalized, truncation, noise_type,
            use_scheduler, use_compressed, max_bd, threads,
        )
        futures[fut] = (t_val, g_name)

    def _abort(sig=None, frame=None):
        print("\nInterrupted - killing workers...", flush=True)
        for f in futures: f.cancel()
        procs = getattr(pool, "_processes", {}) or {}
        for pid in list(procs):
            try: os.kill(pid, signal.SIGKILL)
            except OSError: pass
        os._exit(1)
    signal.signal(signal.SIGINT, _abort)

    t0 = time.time()
    done, failed = 0, 0
    for fut in as_completed(futures):
        t_val, g_name = futures[fut]
        try:
            _, _, loss = fut.result()
            done += 1
            print(f"  ok [{done}/{n_tasks}] time={t_val} gamma={g_name} loss={loss:.6f}",
                  flush=True)
        except Exception as exc:
            failed += 1
            print(f"  FAIL time={t_val} gamma={g_name}: {exc}", flush=True)
    pool.shutdown(wait=True)
    print(f"\nFinished {done}/{n_tasks} ({failed} failed) in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
