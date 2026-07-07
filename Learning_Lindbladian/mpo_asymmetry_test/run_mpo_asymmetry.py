"""Asymmetry test on Learning_MPO_scheduler (random QMLM target + depolarizing noise).

For each (depolarizing_strength p, L_target L) cell:
  • Build a deterministic target QMLM(N, L) with per-site depolarizing channel p.
  • Train 3-layer student PQC twice — once on weight-1 inputs, once on weight-2 —
    using the SAME target params (so both variants chase identical teachers).
  • After each training, re-evaluate the trained model on three input sets:
      weight-1  (24 inputs, full weight-1 basis)
      weight-2  (84 inputs, full weight-2 same-op basis)
      random    (30 random Pauli inputs of weight ∈ [1, N])
  • Save per-sample loss arrays for each (train, eval) cell.

Grid:
  p ∈ {0.01, 0.05}
  L ∈ {3, 4, 5, 6, 7}
  → 10 cells × 2 trainings = 20 trainings.
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
LR     = 0.05
MAX_BD = 32
MAX_ERR = 1e-6
EPOCHS = 200          # both w1 and w2 trained for same epoch count (matched-epoch)
NOISE_TYPE = "depolarizing"
TRUNCATION = False

DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _make_target_param(L_target, p):
    """Deterministic teacher params per (L_target, p). Same seed for both weights."""
    import torch
    g = torch.Generator().manual_seed(20240520 + L_target * 100 + int(round(p * 1000)))
    return (
        torch.rand(L_target, N, 16, dtype=torch.float64, generator=g)
        + 1j * torch.rand(L_target, N, 16, dtype=torch.float64, generator=g)
    )


def _worker(L_target, depol, weight):
    threads = 8
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(threads)
    import torch
    torch.set_num_threads(threads)

    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme

    target_param = _make_target_param(L_target, depol)
    seed = 42 + L_target * 100 + int(round(depol * 1000)) * 10 + weight
    np.random.seed(seed); torch.manual_seed(seed)

    fn_tag = f"N{N}_T{T}_L{L_target}_p{depol}_w{weight}"
    print(f"[PID {os.getpid()}] START  {fn_tag}", flush=True)
    t0 = time.time()
    res = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L_target,
        param_list=target_param,
        depolarizing_strength=depol,
        epochs=EPOCHS, lr=LR,
        normalized=False,
        max_bd=MAX_BD, max_err=MAX_ERR,
        truncation=TRUNCATION, noise_type=NOISE_TYPE,
        use_compressed=False, num_threads=1,
        weight_1_pauli_strings=(weight == 1),
        input_pauli_weight=weight,
    )
    # 11-tuple.
    (model, learning_loss, _, _,
     reported_test_loss, _,
     params_np, p_depolar_np,
     p_dephaseX_np, p_dephaseY_np, p_dephaseZ_np) = res
    train_elapsed = time.time() - t0

    # ── Post-training evaluation on three input sets ────────────────────
    # We re-compute the target via QMLM_output_only.forward_depolarizing_only
    # to match how Learning_MPO_scheduler constructs its target.
    p_depolar_tgt = (torch.ones((L_target, N), dtype=torch.float64)
                     * depol)
    target_module = tdme.QMLM_output_only(
        N, L_target, param=target_param, p_depolar=p_depolar_tgt,
        max_bd=MAX_BD, max_err=MAX_ERR,
    )

    def _eval(input_mps_list):
        losses = []
        with torch.no_grad():
            mpo_fit = model.get_MPO(noise_type=NOISE_TYPE)
            for inp in input_mps_list:
                tgt = target_module.forward_depolarizing_only(
                    inp.copy(), truncation=TRUNCATION,
                )
                for i, ten in enumerate(tgt):
                    ten.reindex_({f'input{i}': f'k{i}'})
                student_out = mpo_fit | inp
                loss = tdme.tensor_network_distance(
                    student_out.astype("complex128"),
                    tgt.astype("complex128"),
                ).item()
                losses.append(loss)
        return np.asarray(losses, dtype=float)

    t1 = time.time()
    arr_w1 = _eval(tdme.Pauli_MPS_weight_1(N))
    arr_w2 = _eval(tdme.Pauli_MPS_weight_2(N))
    np.random.seed(99)
    rand_inputs = [tdme.random_pauli_MPS(N)[0] for _ in range(30)]
    arr_rd = _eval(rand_inputs)
    eval_elapsed = time.time() - t1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    prefix = os.path.join(RESULTS_DIR, fn_tag)
    np.save(f"{prefix}_learning_loss.npy",   np.array(learning_loss))
    np.save(f"{prefix}_eval_weight1.npy",    arr_w1)
    np.save(f"{prefix}_eval_weight2.npy",    arr_w2)
    np.save(f"{prefix}_eval_random.npy",     arr_rd)
    np.save(f"{prefix}_p_depolar.npy",       p_depolar_np)

    print(f"[PID {os.getpid()}] DONE   {fn_tag}  "
          f"w1_eval={arr_w1.mean():.4e}  w2_eval={arr_w2.mean():.4e}  "
          f"rd_eval={arr_rd.mean():.4e}  (train {train_elapsed:.0f}s, eval {eval_elapsed:.0f}s)",
          flush=True)
    return L_target, depol, weight, arr_w1.mean(), arr_w2.mean(), arr_rd.mean()


def main():
    grid = []
    for L in LTARGETS:
        for p in DEPOL_LIST:
            for w in (1, 2):
                grid.append((L, p, w))
    n_tasks = len(grid)
    n_workers = min(n_tasks, 20)
    print(f"  Tasks: {n_tasks}  Workers: {n_workers}", flush=True)
    print(f"  L_targets={LTARGETS}  depols={DEPOL_LIST}", flush=True)

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {pool.submit(_worker, L, p, w): (L, p, w) for L, p, w in grid}

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
            L, p, w = futures[fut]
            try:
                _ = fut.result()
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  L={L} p={p} w={w}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  L={L} p={p} w={w}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)
    print(f"\nFinished {done} tasks ({failed} failed) in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
