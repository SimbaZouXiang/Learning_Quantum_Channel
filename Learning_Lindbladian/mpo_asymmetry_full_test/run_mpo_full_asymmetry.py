"""Full-weight-2 training for the MPO asymmetry experiment.

For each (p, L) cell with p ∈ {0.01, 0.05}, L ∈ {3, 4, 5, 6, 7}:
  - target = QMLM(N=8, L) with per-site depolarizing p (same seed as the
    mpo_asymmetry_test runs, so the teacher is identical)
  - train T_model=3 student on the FULL weight-2 Pauli basis (252 inputs at N=8)
  - epochs = 200 (+ Learning_MPO_scheduler's built-in ~50 fine-tune)
  - evaluate trained model on weight-1 (24), weight-2-same-op (84),
    weight-2-full (252), and 30 random Pauli MPS

Outputs per cell are saved to ./results/ with the tag
  N8_T3_L{L}_p{p}_w2full
"""
import os, sys, signal, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

N        = 8
T        = 3
LR       = 0.05
MAX_BD   = 32
MAX_ERR  = 1e-6
EPOCHS   = 200
NOISE_TYPE = "depolarizing"
TRUNCATION = False

DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _make_target_param(L_target, p):
    """Same target seed as mpo_asymmetry_test/ so the teacher is identical
    across the same-op and full weight-2 runs."""
    import torch
    g = torch.Generator().manual_seed(20240520 + L_target * 100 + int(round(p * 1000)))
    return (
        torch.rand(L_target, N, 16, dtype=torch.float64, generator=g)
        + 1j * torch.rand(L_target, N, 16, dtype=torch.float64, generator=g)
    )


def _worker(L_target, depol):
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
    seed = 42 + L_target * 100 + int(round(depol * 1000)) * 10 + 3
    np.random.seed(seed); torch.manual_seed(seed)

    fn_tag = f"N{N}_T{T}_L{L_target}_p{depol}_w2full"
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
        input_pauli_weight="2_full",
    )
    (model, learning_loss, _, _,
     reported_test_loss, _,
     params_np, p_depolar_np,
     p_dephaseX_np, p_dephaseY_np, p_dephaseZ_np) = res
    train_elapsed = time.time() - t0

    p_depolar_tgt = (torch.ones((L_target, N), dtype=torch.float64) * depol)
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
    arr_w1     = _eval(tdme.Pauli_MPS_weight_1(N))
    arr_w2     = _eval(tdme.Pauli_MPS_weight_2(N))
    arr_w2full = _eval(tdme.Pauli_MPS_weight_2_full(N))
    np.random.seed(99)
    rand_inputs = [tdme.random_pauli_MPS(N)[0] for _ in range(30)]
    arr_rd = _eval(rand_inputs)
    eval_elapsed = time.time() - t1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    prefix = os.path.join(RESULTS_DIR, fn_tag)
    np.save(f"{prefix}_learning_loss.npy",   np.array(learning_loss))
    np.save(f"{prefix}_eval_weight1.npy",    arr_w1)
    np.save(f"{prefix}_eval_weight2.npy",    arr_w2)
    np.save(f"{prefix}_eval_weight2full.npy", arr_w2full)
    np.save(f"{prefix}_eval_random.npy",     arr_rd)
    np.save(f"{prefix}_p_depolar.npy",       p_depolar_np)

    print(f"[PID {os.getpid()}] DONE   {fn_tag}  "
          f"w1={arr_w1.mean():.4e}  w2={arr_w2.mean():.4e}  "
          f"w2full={arr_w2full.mean():.4e}  rd={arr_rd.mean():.4e}  "
          f"(train {train_elapsed:.0f}s, eval {eval_elapsed:.0f}s)", flush=True)
    return L_target, depol, arr_w1.mean(), arr_w2.mean(), arr_w2full.mean(), arr_rd.mean()


def main():
    grid = [(L, p) for L in LTARGETS for p in DEPOL_LIST]
    n_tasks = len(grid)
    n_workers = min(n_tasks, 10)
    print(f"  Tasks: {n_tasks}  Workers: {n_workers}  (252 inputs/training)", flush=True)
    print(f"  L_targets={LTARGETS}  depols={DEPOL_LIST}", flush=True)

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {pool.submit(_worker, L, p): (L, p) for L, p in grid}

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
            L, p = futures[fut]
            try:
                _ = fut.result()
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  L={L} p={p}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  L={L} p={p}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)
    print(f"\nFinished {done} tasks ({failed} failed) in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
