"""A/B compare weight-1 vs weight-2 Pauli training set in Learning_MPO_scheduler.

This mirrors the TDME-target experiment in ../weight1_vs_weight2/ but uses a
much simpler target: a random QMLM with per-site depolarizing noise. Target
parameters are fixed per L_target via a deterministic seed so all variants
at the same L_target chase identical teachers.

Setup
-----
N=8, T=3 PQC student. Target QMLM has L_target ∈ {3, 4, 5, 6, 7} random
parameterized SU(4) gates per pair + per-site depolarizing channel at strength
0.01. Three training variants per L_target:
  w1     : weight-1 inputs (3N=24),  200 + 40 fine-tune epochs
  w2_e60 : weight-2 inputs (84),     60  + 12 fine-tune epochs  (matched sample-evals)
  w2_e200: weight-2 inputs (84),     200 + 40 fine-tune epochs  (matched epochs)

5 L_targets × 3 variants = 15 trainings, run in parallel via ProcessPoolExecutor
inside a single Trillium debugjob node (1h walltime; expected ~10-15 min wall).

Outputs per training:
  results/N8_T3_L{L}_p0.01_{w1|w2_e60|w2_e200}_learning_loss.npy
  results/N8_T3_L{L}_p0.01_{w1|w2_e60|w2_e200}_testing_loss.npy
  results/N8_T3_L{L}_p0.01_{w1|w2_e60|w2_e200}_testing_loss_list.npy
  results/N8_T3_L{L}_target_param.pt  (the teacher parameters; shared across variants)
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
DEPOL  = 0.01
LTARGETS    = [3, 4, 5, 6, 7]
EPOCHS_W1     = 200
EPOCHS_W2_S60 = 60
EPOCHS_W2_S200 = 200
LR          = 0.05
MAX_BD      = 32
MAX_ERR     = 1e-6
TRUNCATION  = False
NOISE_TYPE  = "depolarizing"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _make_target_param(L_target):
    """Deterministic teacher params per L_target."""
    import torch
    g = torch.Generator().manual_seed(20240520 + L_target)
    return (
        torch.rand(L_target, N, 16, dtype=torch.float64, generator=g)
        + 1j * torch.rand(L_target, N, 16, dtype=torch.float64, generator=g)
    )


def _worker(L_target, tag, weight, epochs):
    """One training. tag ∈ {w1, w2_e60, w2_e200} purely for file naming."""
    threads = 8
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(threads)
    import torch
    torch.set_num_threads(threads)

    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme

    target_param = _make_target_param(L_target)
    seed = 42 + L_target * 100 + (1 if weight == 1 else 2 if epochs == 60 else 3)
    np.random.seed(seed); torch.manual_seed(seed)

    fn_tag = f"N{N}_T{T}_L{L_target}_p{DEPOL}_{tag}"
    print(f"[PID {os.getpid()}] START  {fn_tag}  (epochs={epochs} weight={weight})", flush=True)
    t0 = time.time()
    res = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L_target,
        param_list=target_param,
        depolarizing_strength=DEPOL,
        epochs=epochs, lr=LR,
        normalized=False,
        max_bd=MAX_BD, max_err=MAX_ERR,
        truncation=TRUNCATION, noise_type=NOISE_TYPE,
        use_compressed=False, num_threads=1,
        weight_1_pauli_strings=(weight == 1),  # ignored when input_pauli_weight set
        input_pauli_weight=weight,
    )
    # 11-tuple from Learning_MPO_scheduler.
    (model, learning_loss, param_used, p_depolar_used,
     testing_loss, testing_loss_list,
     params_np, p_depolar_np,
     p_dephaseX_np, p_dephaseY_np, p_dephaseZ_np) = res
    elapsed = time.time() - t0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    prefix = os.path.join(RESULTS_DIR, fn_tag)
    np.save(f"{prefix}_learning_loss.npy",     np.array(learning_loss))
    np.save(f"{prefix}_testing_loss.npy",       np.array(float(testing_loss)))
    np.save(f"{prefix}_testing_loss_list.npy", np.array(testing_loss_list))

    # Persist the teacher param once per L_target (overwriting redundantly is fine).
    tp_path = os.path.join(RESULTS_DIR, f"N{N}_T{T}_L{L_target}_target_param.pt")
    if not os.path.exists(tp_path):
        torch.save(target_param, tp_path)

    print(f"[PID {os.getpid()}] DONE   {fn_tag}  testing_loss={float(testing_loss):.5e}  "
          f"elapsed={elapsed:.1f}s", flush=True)
    return L_target, tag, float(testing_loss), elapsed


def main():
    grid = []
    for L in LTARGETS:
        grid.append((L, "w1",      1, EPOCHS_W1))
        grid.append((L, "w2_e60",  2, EPOCHS_W2_S60))
        grid.append((L, "w2_e200", 2, EPOCHS_W2_S200))
    n_tasks = len(grid)

    n_workers = min(n_tasks, 15)
    print("=" * 60, flush=True)
    print(f"  Tasks: {n_tasks}  Workers: {n_workers}", flush=True)
    print(f"  L_targets: {LTARGETS}", flush=True)
    print(f"  depolarizing_strength: {DEPOL}", flush=True)
    print("=" * 60, flush=True)

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {pool.submit(_worker, L, tag, w, e): (L, tag, w, e) for L, tag, w, e in grid}

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
            L, tag, w, e = futures[fut]
            try:
                _, _, loss, secs = fut.result()
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  L={L} tag={tag:8s}  "
                      f"testing_loss={loss:.4e}  ({secs:.0f}s)", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  L={L} tag={tag}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"\nFinished {done} tasks ({failed} failed) in {elapsed:.0f}s "
          f"({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
