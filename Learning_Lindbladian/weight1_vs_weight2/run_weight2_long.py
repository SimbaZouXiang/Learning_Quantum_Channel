"""Follow-up to run_weight1_vs_weight2.py: train weight-2 at the SAME epoch
count as weight-1 (200 + 40 fine-tune = 240 total), so the comparison is no
longer biased against weight-2 by under-training.

Results saved with a `_e200` suffix so they don't overwrite the 60-epoch ones
already on disk; the plot script will overlay both.
"""

import os
import sys
import signal
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

N      = 8
T      = 3
mu     = 1
J      = 1
gamma  = [0.1] * N
t_eval = 1.0
LTARGETS    = [3, 4, 5, 6, 7]
EPOCHS_W2   = 200            # matched to weight-1
LR          = 0.05
MAX_BD      = 64
MAX_ERR     = 1e-8
NOISE_TYPE  = "dephasing"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
SUFFIX      = "_e200"        # distinguishes from the prior 60-epoch run


def _worker(L_target, epochs):
    threads = 8
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(threads)
    import torch
    torch.set_num_threads(threads)

    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme

    seed = 42 + L_target * 100 + 2     # match the seed used for weight-2 in the original run
    np.random.seed(seed); torch.manual_seed(seed)

    tag = f"N{N}_T{T}_L{L_target}_t{t_eval}_g{gamma[0]}_w2{SUFFIX}"
    print(f"[PID {os.getpid()}] START  {tag} (epochs={epochs} threads={threads})",
          flush=True)
    t0 = time.time()
    result = tdme.Learning_TDME_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L_target,
        mu=mu, gamma=gamma, J=J, t=t_eval,
        epochs=epochs, lr=LR,
        normalized=False,
        max_bd=MAX_BD, max_err=MAX_ERR,
        truncation=True, noise_type=NOISE_TYPE,
        use_scheduler=False, use_compressed=False,
        num_threads=1, data_dir=None,
        input_pauli_weight=2,
    )
    elapsed = time.time() - t0
    model, learning_loss, testing_loss, testing_loss_list = result

    os.makedirs(RESULTS_DIR, exist_ok=True)
    prefix = os.path.join(RESULTS_DIR, tag)
    np.save(f"{prefix}_learning_loss.npy",     np.array(learning_loss))
    np.save(f"{prefix}_testing_loss.npy",       np.array(float(testing_loss)))
    np.save(f"{prefix}_testing_loss_list.npy", np.array(testing_loss_list))

    print(f"[PID {os.getpid()}] DONE   {tag}  testing_loss={float(testing_loss):.5e}  "
          f"elapsed={elapsed:.1f}s", flush=True)
    return L_target, float(testing_loss), elapsed


def main():
    n_tasks = len(LTARGETS)
    n_workers = n_tasks
    print(f"  Tasks: {n_tasks} (weight-2 only at epochs={EPOCHS_W2})", flush=True)
    print(f"  L_targets: {LTARGETS}", flush=True)

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {pool.submit(_worker, L, EPOCHS_W2): L for L in LTARGETS}

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
            L = futures[fut]
            try:
                _, loss, secs = fut.result()
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  L={L}  "
                      f"testing_loss={loss:.4e}  ({secs:.0f}s)", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  L={L}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"\nFinished {done} tasks ({failed} failed) in {elapsed:.0f}s "
          f"({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
