"""PQC training run for target_time = 3.0, 10 gamma values in parallel.
Mirrors Learning_TDME_using_data_2.py exactly except for target_times."""
import os, sys, signal, argparse, time
import numpy as np
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTAL_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def _worker(target_time, gamma_name, N, T, model_to_learn_layer,
            mu, J, epochs, lr, normalized, truncation, noise_type,
            use_scheduler, use_compressed, max_bd, threads):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch
    torch.set_num_threads(threads)

    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    import TDME_Trott as tdme

    gamma = [gamma_name * 0.01] * N
    data_dir = os.path.join(
        SCRIPT_DIR, "Learning_data",
        f"N{N}_T{model_to_learn_layer}_mu{float(mu)}_gamma{round(gamma[0], 2)}_t{target_time}",
    )

    tag = f"time={target_time}, gamma_name={gamma_name}"
    print(f"[PID {os.getpid()}] START  {tag}  (threads={threads})", flush=True)

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

    prefix = (
        f"Learning_result/TDME_N{N}_T{T}_Modeltolearnlayer{model_to_learn_layer}"
        f"_time{target_time}_gamma{gamma_name:03d}"
    )
    np.save(f"{prefix}_learning_loss.npy",     np.array(learning_loss))
    np.save(f"{prefix}_testing_loss.npy",       np.array(testing_loss))
    np.save(f"{prefix}_testing_loss_list.npy", np.array(testing_loss_list))

    print(f"[PID {os.getpid()}] DONE   {tag}  testing_loss={testing_loss}", flush=True)
    return target_time, gamma_name, float(testing_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    cli = ap.parse_args()

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

    target_times = [5.0]
    gamma_names  = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]

    grid = [(t, g) for t in target_times for g in gamma_names]
    n_tasks = len(grid)

    n_workers = cli.workers or min(n_tasks, 10)
    n_workers = min(n_workers, n_tasks, TOTAL_CORES)
    threads   = min(19, max(1, TOTAL_CORES // n_workers))

    print("=" * 60)
    print(f"  CPU cores      : {TOTAL_CORES}")
    print(f"  Grid points    : {n_tasks}  ({len(target_times)} times × {len(gamma_names)} gammas)")
    print(f"  Workers        : {n_workers}")
    print(f"  Threads/worker : {threads}")
    print("=" * 60, flush=True)

    ctx  = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {}
    for t_val, g_name in grid:
        fut = pool.submit(
            _worker, t_val, g_name, N, T, model_to_learn_layer,
            mu, J, epochs, lr, normalized, truncation, noise_type,
            use_scheduler, use_compressed, max_bd, threads,
        )
        futures[fut] = (t_val, g_name)

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
            t_val, g_name = futures[fut]
            try:
                _, _, loss = fut.result()
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  time={t_val}, gamma_name={g_name}  loss={loss:.6f}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  time={t_val}, gamma_name={g_name}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)
    print(f"\nFinished {done} tasks ({failed} failed) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
