"""Peak-location sweep: empirical L_max(p) with seeds and error bars.

Trains the paper's N=8 model (T in {2,3}) against Haar-random depolarizing
targets over a log-spaced p grid and L in 2..10, with SEVERAL seeds per point
(seed controls both the Haar target and the optimizer init), so that:
  * the empirical peak location L_max(p, T) can be extracted with error bars,
  * the p^(-1/2) scaling prediction of Sec. VII can be fit directly,
  * multi-seed error bars can be added to the N=8 figures.

One point == one Learning_MPO_scheduler run with the exact paper protocol
(epochs=200 + epochs//5 fine-tune, lr=0.05, noise_type="depolarizing",
no truncation at N=8).  Points already on disk are skipped (safe to requeue).

Subsets:
  --subset debug : T=3, p in {0.005,0.01,0.02,0.03,0.05}, L 2..8, seeds {0,1}
                   (70 points; fits a 1 h debug job on a full node)
  --subset full  : T in {2,3}, p in {0.002,...,0.05} (9 values), L 2..10,
                   seeds {0,1,2}  (486 points; a few hours on a full node)
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

P_FULL = [0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
P_DEBUG = [0.005, 0.01, 0.02, 0.03, 0.05]


def tag(p):
    return f"{int(round(p * 10000)):04d}"  # p in 1e-4 units: 0.002 -> 0020


def point_prefix(outdir, N, T, L, p, seed):
    return os.path.join(outdir, f"PeakSweep_N{N}_T{T}_L{L}_p{tag(p)}_seed{seed}")


def run_single_point(args):
    (N, T, L, p, seed, epochs, lr, threads, outdir) = args
    os.makedirs(outdir, exist_ok=True)
    prefix = point_prefix(outdir, N, T, L, p, seed)
    if os.path.exists(prefix + "_testing_loss.npy"):
        return (T, p, L, seed, "skip", 0.0)

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMBA_NUM_THREADS"] = "1"

    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, SCRIPT_DIR)
    from qcl import Learning_MPO_scheduler

    torch.manual_seed(seed)
    np.random.seed(seed)

    (model, learning_loss, target_param, p_depolar,
     testing_loss, testing_loss_list, model_param, model_p_depolar,
     model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ
     ) = Learning_MPO_scheduler(
        N, T, L,
        depolarizing_strength=p,
        epochs=epochs, lr=lr,
        normalized=False, truncation=False,
        noise_type="depolarizing",
        num_threads=None,          # sequential inner loop; parallelism is across points
        use_compressed=False,
    )

    np.save(prefix + "_learning_loss.npy", np.array(learning_loss))
    np.save(prefix + "_testing_loss.npy", np.array(testing_loss))
    np.save(prefix + "_testing_loss_list.npy", np.array(testing_loss_list))
    np.save(prefix + "_model_p_depolar.npy", np.array(model_p_depolar))
    np.save(prefix + "_target_param.npy", np.array(target_param))
    final = float(np.array(learning_loss)[-1]) if len(learning_loss) else float("nan")
    return (T, p, L, seed, "done", final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["debug", "full"], default="full")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads-per-worker", type=int, default=8)
    ap.add_argument("--outdir", type=str, default=os.path.join(SCRIPT_DIR, "Result_peak_sweep"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.subset == "debug":
        grid = [(args.N, 3, L, p, s, args.epochs, args.lr, args.threads_per_worker, args.outdir)
                for p in P_DEBUG for L in range(2, 9) for s in (0, 1)]
    else:
        grid = [(args.N, T, L, p, s, args.epochs, args.lr, args.threads_per_worker, args.outdir)
                for T in (2, 3) for p in P_FULL for L in range(2, 11) for s in (0, 1, 2)]

    print(f"peak sweep subset={args.subset}: {len(grid)} points, "
          f"{args.workers} workers x {args.threads_per_worker} threads", flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futs = {pool.submit(run_single_point, g): g for g in grid}
        for f in as_completed(futs):
            g = futs[f]
            try:
                T, p, L, seed, status, final = f.result()
                done += 1
                print(f"[{done}/{len(grid)}] T={T} p={p} L={L} seed={seed}: {status} "
                      f"final_loss={final:.5f}", flush=True)
            except Exception as exc:
                done += 1
                print(f"[{done}/{len(grid)}] FAIL {g[:5]}: {exc}", flush=True)
    print("peak sweep finished", flush=True)


if __name__ == "__main__":
    main()
