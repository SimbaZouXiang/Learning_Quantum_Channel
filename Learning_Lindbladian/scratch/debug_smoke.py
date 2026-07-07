"""Small-system smoke test of the training driver (same code path as
Learning_TDME_using_data_*.py, but N=10 / 2 grid points / 10 epochs) so we
can confirm the 10-worker × 19-thread config launches, trains, and tests
without OOM before committing the full 24-h jobs."""

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
    print(f"[PID {os.getpid()}] DONE  {tag}  loss0={learning_loss[0]:.4f} "
          f"loss_final={learning_loss[-1]:.4f}  testing_loss={testing_loss:.4f}  "
          f"wall={wall:.1f}s", flush=True)
    return target_time, gamma_name, float(testing_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    cli = ap.parse_args()

    N                    = 10
    T                    = 3
    model_to_learn_layer = 3
    mu                   = 1
    J                    = 1
    epochs               = 10
    lr                   = 0.05
    normalized           = False
    truncation           = True
    noise_type           = "dephasing"
    use_scheduler        = False
    use_compressed       = False
    max_bd               = 64

    target_times = [0.2]
    gamma_names  = [0, 2]  # both have cached N=10 data
    grid = [(t, g) for t in target_times for g in gamma_names]
    n_tasks = len(grid)

    threads_per_worker_target = 19
    n_workers = cli.workers or min(n_tasks, max(1, TOTAL_CORES // threads_per_worker_target))
    n_workers = min(n_workers, n_tasks, TOTAL_CORES)
    threads   = max(1, TOTAL_CORES // n_workers)

    print("=" * 60)
    print(f"  CPU cores      : {TOTAL_CORES}")
    print(f"  Grid points    : {n_tasks}")
    print(f"  Workers        : {n_workers}")
    print(f"  Threads/worker : {threads}")
    print("=" * 60, flush=True)

    if _HAS_LOKY:
        print("Using loky reusable executor.", flush=True)
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

    t0 = time.time()
    done, failed = 0, 0
    for fut in as_completed(futures):
        t_val, g_name = futures[fut]
        try:
            _, _, loss = fut.result()
            done += 1
            print(f"  ✓ [{done}/{n_tasks}] time={t_val} gamma={g_name} loss={loss:.6f}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"  ✗ time={t_val} gamma={g_name}: {exc}", flush=True)
    pool.shutdown(wait=True)
    print(f"\nSMOKE {done}/{n_tasks} ok, {failed} failed  total_wall={time.time()-t0:.1f}s", flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
