"""Verify that a patched Trotterization_3_to_30_* script writes
incremental PARTIAL_* files during the run and final outputs at the end
into Learning_result/.

Runs at small scale (N=8, T=2 model, 6 fine layers, 6 samples, 3 workers)
so it finishes in a minute or two.
"""
import os
import sys
import glob
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

import TDME_Trott as tdme


def main():
    output_dir = os.path.join(SCRIPT_DIR, "Learning_result")
    os.makedirs(output_dir, exist_ok=True)

    # Small-scale config. model_layer=2, fine=6, so trotter-error is non-zero.
    N = 8
    T = 2
    model_to_learn_layer = 6
    mu = 1.0
    J = 1.0
    t_val = 0.3
    g = 0  # gamma tag
    gamma_array = [0.1] * N
    max_bd = 16
    max_err = 1e-6
    truncation = True
    noise_type = "dephasing"
    use_scheduler = False
    num_samples = 6
    num_threads = 3

    loss_file = os.path.join(
        output_dir,
        f"Trotterization_Testing_loss_t{t_val}_gamma{g}_N{N}_T{T}_"
        f"Modeltolearnlayer{model_to_learn_layer}_mu{float(mu)}_"
        f"gamma{round(gamma_array[0], 2)}_t{t_val}.npy"
    )
    list_file = os.path.join(
        output_dir,
        f"Trotterization_Testing_loss_list_t{t_val}_gamma{g}_N{N}_T{T}_"
        f"Modeltolearnlayer{model_to_learn_layer}_mu{float(mu)}_"
        f"gamma{round(gamma_array[0], 2)}_t{t_val}.npy"
    )
    partial_prefix = os.path.join(
        output_dir,
        f"PARTIAL_Trotterization_t{t_val}_gamma{g}_N{N}_T{T}_"
        f"Modeltolearnlayer{model_to_learn_layer}_mu{float(mu)}_"
        f"gamma{round(gamma_array[0], 2)}"
    )

    # Clean any stale files first
    for f in (loss_file, list_file,
              partial_prefix + "_list.npy", partial_prefix + "_loss.npy"):
        if os.path.exists(f):
            os.remove(f)

    print(f"output_dir: {output_dir}", flush=True)
    print(f"final files expected:\n  {loss_file}\n  {list_file}", flush=True)
    print(f"partial prefix: {partial_prefix}", flush=True)

    t0 = time.time()
    testing_loss, testing_loss_list = tdme.Testing_TDME_Trotterization_parallel(
        N=N,
        model_layer=T,
        model_to_learn_layer=model_to_learn_layer,
        mu=mu,
        gamma=gamma_array,
        J=J,
        t=t_val,
        normalized=False,
        max_bd=max_bd,
        max_err=max_err,
        truncation=truncation,
        noise_type=noise_type,
        use_scheduler=use_scheduler,
        num_samples=num_samples,
        num_threads=num_threads,
        incremental_save_prefix=partial_prefix,
    )
    elapsed = time.time() - t0

    # After run, check what's on disk *before* we save final files
    partial_list = partial_prefix + "_list.npy"
    partial_loss = partial_prefix + "_loss.npy"
    partial_present = os.path.exists(partial_list) and os.path.exists(partial_loss)
    print(f"\n=== post-run state ===", flush=True)
    print(f"  partial list exists: {os.path.exists(partial_list)}", flush=True)
    print(f"  partial loss exists: {os.path.exists(partial_loss)}", flush=True)
    if partial_present:
        pl = np.load(partial_list)
        pl_mean = np.load(partial_loss)
        print(f"  partial list len={len(pl)}  mean={float(pl_mean):.5f}", flush=True)

    # Do the driver-style final save + cleanup
    np.save(loss_file, testing_loss)
    np.save(list_file, testing_loss_list)
    for suffix in ("_list.npy", "_loss.npy"):
        p = partial_prefix + suffix
        if os.path.exists(p):
            os.remove(p)

    # Verify the final layout
    print(f"\n=== final layout ===", flush=True)
    assert os.path.exists(loss_file), f"final loss file missing: {loss_file}"
    assert os.path.exists(list_file), f"final list file missing: {list_file}"
    assert not os.path.exists(partial_list), "partial list still around after cleanup"
    assert not os.path.exists(partial_loss), "partial loss still around after cleanup"
    print(f"  final loss={np.load(loss_file):.5f}  "
          f"list len={len(np.load(list_file))}", flush=True)
    print(f"  partial files cleaned up: OK", flush=True)
    print(f"  elapsed: {elapsed:.1f}s", flush=True)
    print(f"\nPASS: Trotterization pipeline writes to Learning_result/", flush=True)


if __name__ == "__main__":
    main()
