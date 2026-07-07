"""Rerun the w1-only and combined-(w1+w2-full) MPO trainings, saving ALL
student parameters this time so we can later reconstruct the trained
channels (e.g., for MPO-MPO Frobenius/Choi-distance comparisons).

For each (L, p, variant):
  - train Learning_MPO_scheduler with the SAME target seed as before so
    the teacher is identical to the prior runs
  - save:
      *_state_dict.pt        full PyTorch model state (gold standard)
      *_params.npy           (T, N, 16) complex SU(4) gate params
      *_p_depolar.npy        (T, N) depolarizing rates
      *_p_dephaseX.npy       (T, N) X-dephasing rates  (untrained in this run)
      *_p_dephaseY.npy       (T, N) Y-dephasing rates
      *_p_dephaseZ.npy       (T, N) Z-dephasing rates
      *_target_param.pt      teacher's (L, N, 16) param tensor
      *_learning_loss.npy    full training trajectory
      *_eval_weight1.npy     per-sample loss on the 24 weight-1 basis
      *_eval_weight2.npy     per-sample loss on the 84 same-op weight-2 basis
      *_eval_random.npy      per-sample loss on 30 random Paulis
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
VARIANTS = [("w1", 1), ("combined", "combined")]
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _make_target_param(L_target, p):
    """Same seed as the prior mpo_asymmetry_test / mpo_combined_test runs."""
    import torch
    g = torch.Generator().manual_seed(20240520 + L_target * 100 + int(round(p * 1000)))
    return (
        torch.rand(L_target, N, 16, dtype=torch.float64, generator=g)
        + 1j * torch.rand(L_target, N, 16, dtype=torch.float64, generator=g)
    )


def _worker(L_target, depol, variant_tag, variant_arg):
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
    seed_map = {"w1": 1, "combined": 4}
    seed = 42 + L_target * 100 + int(round(depol * 1000)) * 10 + seed_map[variant_tag]
    np.random.seed(seed); torch.manual_seed(seed)

    fn_tag = f"N{N}_T{T}_L{L_target}_p{depol}_{variant_tag}"
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
        input_pauli_weight=variant_arg,
    )
    (model, learning_loss, _, _,
     reported_test_loss, _,
     params_np, p_depolar_np,
     p_dephaseX_np, p_dephaseY_np, p_dephaseZ_np) = res
    train_elapsed = time.time() - t0

    # Eval on three input sets so future analysis can sanity-check vs the prior runs.
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
    arr_w1 = _eval(tdme.Pauli_MPS_weight_1(N))
    arr_w2 = _eval(tdme.Pauli_MPS_weight_2(N))
    np.random.seed(99)
    rand_inputs = [tdme.random_pauli_MPS(N)[0] for _ in range(30)]
    arr_rd = _eval(rand_inputs)
    eval_elapsed = time.time() - t1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    prefix = os.path.join(RESULTS_DIR, fn_tag)
    # FULL parameter dump
    torch.save(model.state_dict(), f"{prefix}_state_dict.pt")
    torch.save(target_param,         f"{prefix}_target_param.pt")
    np.save(f"{prefix}_params.npy",      params_np)
    np.save(f"{prefix}_p_depolar.npy",   p_depolar_np)
    np.save(f"{prefix}_p_dephaseX.npy",  p_dephaseX_np)
    np.save(f"{prefix}_p_dephaseY.npy",  p_dephaseY_np)
    np.save(f"{prefix}_p_dephaseZ.npy",  p_dephaseZ_np)
    np.save(f"{prefix}_learning_loss.npy", np.array(learning_loss))
    np.save(f"{prefix}_eval_weight1.npy",   arr_w1)
    np.save(f"{prefix}_eval_weight2.npy",   arr_w2)
    np.save(f"{prefix}_eval_random.npy",    arr_rd)

    print(f"[PID {os.getpid()}] DONE   {fn_tag}  "
          f"w1={arr_w1.mean():.4e} w2={arr_w2.mean():.4e} rd={arr_rd.mean():.4e}  "
          f"(train {train_elapsed:.0f}s, eval {eval_elapsed:.0f}s)", flush=True)
    return L_target, depol, variant_tag


def main():
    grid = []
    for L in LTARGETS:
        for p in DEPOL_LIST:
            for vt, va in VARIANTS:
                grid.append((L, p, vt, va))
    n_tasks = len(grid)
    n_workers = min(n_tasks, 20)
    print(f"  Tasks: {n_tasks}  Workers: {n_workers}", flush=True)
    print(f"  L_targets={LTARGETS}  depols={DEPOL_LIST}  variants={[v[0] for v in VARIANTS]}",
          flush=True)

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    futures = {pool.submit(_worker, L, p, vt, va): (L, p, vt) for L, p, vt, va in grid}

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
            L, p, vt = futures[fut]
            try:
                _ = fut.result()
                done += 1
                print(f"  ✓  [{done}/{n_tasks}]  L={L} p={p} {vt}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  ✗  L={L} p={p} {vt}: {exc}", flush=True)
    finally:
        pool.shutdown(wait=True)
    print(f"\nFinished {done} ({failed} failed) in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
