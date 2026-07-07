"""
Generate and cache MPS training data for Learning_unitary.

For each L_target (3–10):
  - Generate one random parameter set and save it
  - For each p_depolar in {0.00, 0.02, 0.04, 0.06, 0.08, 0.10}:
      - Call get_input_and_output_MPS to produce 3N output MPS
      - Save each output MPS to an individual .npz file in training_data/

Input MPS (weight-1 Pauli) are deterministic and saved once.

Usage:
    python generate_training_data.py
    python generate_training_data.py --workers 4
"""

import os
import sys
import signal
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# ── Determine available cores ──────────────────────────────────────────
TOTAL_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK",
                  os.cpu_count() or 1))

# Output directory for all MPS data files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


def run_single_task(args):
    """Generate and save MPS data for one (L_target, p_depolar) combination.

    Each worker limits its own thread usage so that the total across all
    workers does not exceed the available cores.
    """
    (L_target, p_depolar_value, N, param_np, data_dir,
     max_bd, max_err, truncation, threads_per_worker) = args

    # ── Limit threads BEFORE importing heavy libraries ──
    os.environ["NUMBA_NUM_THREADS"]    = str(threads_per_worker)
    os.environ["OMP_NUM_THREADS"]      = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"]      = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    import torch
    torch.set_num_threads(threads_per_worker)

    # Make the TDME_Trott shim (in the sibling Learning_Lindbladian/) and this
    # script's own directory importable regardless of CWD.
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _p in (parent_dir,
               os.path.join(parent_dir, "Learning_Lindbladian"),
               os.path.dirname(os.path.abspath(__file__))):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import TDME_Trott as tdme
    from load_training_data import save_mps

    # Convert param back to torch tensor
    param = torch.tensor(param_np)

    # Build p_depolar matrix: shape (L_target, N)
    p_depolar = torch.ones((L_target, N), dtype=torch.float64) * p_depolar_value

    # Format p_depolar for filename: e.g. 0.02 -> "002"
    p_str = f"{p_depolar_value:.2f}".replace(".", "")

    print(f"[Worker PID {os.getpid()}] L={L_target}, p={p_depolar_value}, "
          f"threads={threads_per_worker}", flush=True)

    # ── Generate training data ─────────────────────────────────────────
    MPS_weight1, target_mps_list, param = tdme.get_input_and_output_MPS(
        N, L_target, param=param, p_depolar=p_depolar,
        max_bd=max_bd, max_err=max_err,
        truncation=truncation, noise_type="depolarizing",
        num_threads=1,
    )

    # ── Save output MPS ────────────────────────────────────────────────
    num_mps = len(target_mps_list)  # should be 3*N
    for mps_id, mps in enumerate(target_mps_list, start=1):
        filename = f"Unitary_output_N_{N}_L_{L_target}_p_{p_str}_MPS{mps_id}.npz"
        filepath = os.path.join(data_dir, filename)
        save_mps(mps, filepath)

    print(f"[Worker PID {os.getpid()}] DONE — L={L_target}, p={p_depolar_value}, "
          f"saved {num_mps} output MPS", flush=True)

    return L_target, p_depolar_value, num_mps


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save MPS training data for Learning_unitary"
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    args = parser.parse_args()

    # ── Problem parameters ─────────────────────────────────────────────
    N = 30
    L_targets = list(range(3, 11))          # 3, 4, 5, 6, 7, 8, 9, 10
    p_depolar_values = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    max_bd = 256
    max_err = 1E-10
    truncation = True

    # ── Create output directory ────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Generate and save random parameters for each L_target ──────────
    # Also save input MPS once (they are deterministic for given N)
    print("=" * 60)
    print("Generating random parameters for each L_target...")

    # Need torch for parameter generation
    import torch

    params_dict = {}  # L_target -> numpy array
    for L in L_targets:
        param = (torch.rand(L, N, 16, dtype=torch.float64)
                 + 1j * torch.rand(L, N, 16, dtype=torch.float64))
        param_np = param.numpy()
        params_dict[L] = param_np
        param_file = os.path.join(DATA_DIR, f"params_L{L}.npy")
        np.save(param_file, param_np)
        print(f"  Saved parameters for L={L}: shape {param_np.shape}")

    # ── Save input MPS (weight-1 Pauli, deterministic) ─────────────────
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _p in (parent_dir,
               os.path.join(parent_dir, "Learning_Lindbladian"),
               os.path.dirname(os.path.abspath(__file__))):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import TDME_Trott as tdme
    from load_training_data import save_mps

    print(f"Generating and saving input MPS (weight-1 Pauli) for N={N}...")
    MPS_weight1 = tdme.Pauli_MPS_weight_1(N)
    for mps_id, mps in enumerate(MPS_weight1, start=1):
        filename = f"Unitary_input_N_{N}_MPS{mps_id}.npz"
        filepath = os.path.join(DATA_DIR, filename)
        save_mps(mps, filepath)
    print(f"  Saved {len(MPS_weight1)} input MPS")

    # ── Build task list ────────────────────────────────────────────────
    tasks = []
    for L in L_targets:
        for p in p_depolar_values:
            tasks.append((L, p, N, params_dict[L], DATA_DIR,
                          max_bd, max_err, truncation))

    num_tasks = len(tasks)
    num_workers = args.workers if args.workers else min(num_tasks, 6)
    num_workers = min(num_workers, num_tasks, TOTAL_CORES)

    # Give each worker a fair share of cores
    threads_per_worker = min(16, max(1, TOTAL_CORES // num_workers))

    # Append threads_per_worker to each task tuple
    tasks = [t + (threads_per_worker,) for t in tasks]

    print("=" * 60)
    print(f"Total cores available  : {TOTAL_CORES}")
    print(f"Number of tasks        : {num_tasks}")
    print(f"Number of workers      : {num_workers}")
    print(f"Threads per worker     : {threads_per_worker}")
    print(f"L_targets              : {L_targets}")
    print(f"p_depolar values       : {p_depolar_values}")
    print(f"Output directory       : {DATA_DIR}")
    print("=" * 60, flush=True)

    t0 = time.time()

    # ── Launch workers ─────────────────────────────────────────────────
    from multiprocessing import get_context
    ctx = get_context("spawn")

    pool = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
    futures = {pool.submit(run_single_task, task): task for task in tasks}

    # ── Install signal handler so Ctrl+C kills everything ──────────────
    def _kill_all_workers(signum=None, frame=None):
        print("\n⚠ Ctrl+C received — force-killing all workers...", flush=True)
        for f in futures:
            f.cancel()
        for pid in list(pool._processes.keys()):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        os._exit(1)

    signal.signal(signal.SIGINT, _kill_all_workers)

    # ── Collect results ────────────────────────────────────────────────
    completed = 0
    try:
        for future in as_completed(futures):
            task = futures[future]
            L_target, p_val = task[0], task[1]
            try:
                L_done, p_done, n_mps = future.result()
                completed += 1
                print(f"✓ [{completed}/{num_tasks}] Completed: L={L_done}, "
                      f"p={p_done:.2f}, saved {n_mps} MPS", flush=True)
            except Exception as exc:
                completed += 1
                print(f"✗ [{completed}/{num_tasks}] FAILED: L={L_target}, "
                      f"p={p_val:.2f}: {exc}", flush=True)
    except KeyboardInterrupt:
        _kill_all_workers()
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"\nAll tasks finished in {elapsed:.1f} s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
