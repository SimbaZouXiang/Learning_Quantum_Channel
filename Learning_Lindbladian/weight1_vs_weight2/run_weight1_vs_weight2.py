"""A/B compare weight-1 vs weight-2 Pauli training set in Learning_TDME_scheduler.

Setup
-----
At N=8 the two training-set sizes are
  weight-1:  3 * N            = 24    samples
  weight-2:  3 * N*(N-1)/2    = 84    samples   (3.5x the weight-1 count)
so we offset the epoch counts to make the TOTAL number of sample evaluations
roughly equal across the two variants:
  weight-1:  epochs = 200   ( + 40 fine-tune)  → 24 * 240 ≈ 5760 evals
  weight-2:  epochs =  60   ( + 12 fine-tune)  → 84 *  72 ≈ 6048 evals
(`Learning_TDME_scheduler` runs `epochs // 5` fine-tune epochs after the main loop.)

Target circuit
--------------
The same target TDME Lindbladian is used for both variants at each L_target.
The target depends only on (N, L_target, mu, gamma, J, t) — there is no
random `param` argument — so weight-1 and weight-2 students at fixed L_target
are chasing identical teachers.

We sweep L_target ∈ {3, 4, 5, 6, 7}; T_model = 3; fixed (mu=1, gamma=0.1, J=1, t=1.0).

Outputs
-------
For each (L_target, weight), we save to ./results/:
  N8_T3_L{L}_t1.0_g0.1_w{1|2}_learning_loss.npy
  N8_T3_L{L}_t1.0_g0.1_w{1|2}_testing_loss.npy
  N8_T3_L{L}_t1.0_g0.1_w{1|2}_testing_loss_list.npy
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

# ── fixed problem parameters ────────────────────────────────────────
N      = 8
T      = 3
mu     = 1
J      = 1
gamma  = [0.1] * N        # physical γ = 0.1 per site, on-site Z-dephasing
t_eval = 1.0
LTARGETS    = [3, 4, 5, 6, 7]
EPOCHS_W1   = 200          # → 240 with fine-tune
EPOCHS_W2   = 60           # → 72  with fine-tune
LR          = 0.05
MAX_BD      = 64
MAX_ERR     = 1e-8
NOISE_TYPE  = "dephasing"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _worker(L_target, weight, epochs):
    """One training. Runs in a fresh process so BLAS threads can be pinned."""
    threads = 8
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(threads)
    import torch
    torch.set_num_threads(threads)

    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme

    # Deterministic per-(L, weight) seeds so re-runs reproduce.
    seed = 42 + L_target * 100 + weight
    np.random.seed(seed); torch.manual_seed(seed)

    tag = f"N{N}_T{T}_L{L_target}_t{t_eval}_g{gamma[0]}_w{weight}"
    print(f"[PID {os.getpid()}] START  {tag} (epochs={epochs} threads={threads})", flush=True)

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
        input_pauli_weight=weight,
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
    return L_target, weight, float(testing_loss), elapsed


def main():
    # Build (L_target, weight, epochs) grid.
    grid = []
    for L in LTARGETS:
        grid.append((L, 1, EPOCHS_W1))
        grid.append((L, 2, EPOCHS_W2))
    n_tasks = len(grid)

    # Use up to 10 workers in parallel (one per (L, weight)). With 192 cores on
    # a debugjob node and 8 threads per worker, total BLAS thread count ≤ 80 —
    # well within the node's CPU budget.
    n_workers = min(n_tasks, 10)
    print("=" * 60, flush=True)
    print(f"  Tasks: {n_tasks}  Workers: {n_workers}", flush=True)
    print(f"  L_targets: {LTARGETS}", flush=True)
    print(f"  weight-1 epochs={EPOCHS_W1} (+{EPOCHS_W1//5} fine-tune); samples=3N=24",
          flush=True)
    print(f"  weight-2 epochs={EPOCHS_W2} (+{EPOCHS_W2//5} fine-tune); samples=3N(N-1)/2=84",
          flush=True)
    print("=" * 60, flush=True)

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {
        pool.submit(_worker, L, w, e): (L, w, e) for L, w, e in grid
    }

    def _abort(sig=None, frame=None):
        print("\nINTERRUPT — killing workers", flush=True)
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
            L, w, e = futures[fut]
            try:
                _, _, loss, secs = fut.result()
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  L={L} w={w}  "
                      f"testing_loss={loss:.4e}  ({secs:.0f}s)", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  L={L} w={w}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"\nFinished {done} tasks ({failed} failed) in {elapsed:.0f}s "
          f"({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
