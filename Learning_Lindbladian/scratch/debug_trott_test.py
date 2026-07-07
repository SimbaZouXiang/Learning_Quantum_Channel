"""Run Testing_TDME_Trotterization_parallel exactly as the saved jobs did,
at small N with T=3 vs L=30.  If the result is still ~1e-13 medians, the bug
is in the test wrapper.  If it's ~0.1, the bug is somewhere in the older code
that produced the saved files (and the current code is fine).
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Casting complex")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import torch
torch.set_num_threads(1)

import TDME_Trott as tdme

if __name__ == "__main__":
    N = 30        # match the saved-job regime
    T_model = 3
    T_target = 30
    t_val = 0.8
    gamma_val = 0.1

    print(f"N={N}, T_model={T_model}, T_target={T_target}, "
          f"t={t_val}, gamma={gamma_val}", flush=True)

    testing_loss, testing_loss_list = tdme.Testing_TDME_Trotterization_parallel(
        N=N,
        model_layer=T_model,
        model_to_learn_layer=T_target,
        mu=1.0,
        gamma=[gamma_val] * N,
        J=1.0,
        t=t_val,
        normalized=False,
        max_bd=64,
        max_err=1e-8,
        truncation=True,
        noise_type="dephasing",
        use_scheduler=False,
        num_samples=300,
        num_threads=48,
    )
    arr = np.asarray(testing_loss_list, dtype=float)
    print()
    print(f"  list mean   : {arr.mean():.4e}")
    print(f"  list median : {np.median(arr):.4e}")
    print(f"  list min    : {arr.min():.4e}")
    print(f"  list max    : {arr.max():.4e}")
    print(f"  scalar      : {testing_loss:.4e}")
    print(f"  first 10    : {arr[:10]}")
