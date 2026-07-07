"""Direct sanity check: run T=3 and T=30 Trotter on the SAME random Pauli
input at the (t=0.8, gamma=0.1) regime where the saved testing list has
median ~1e-13, and inspect the actual outputs.

If the 3-layer and 30-layer Trotter genuinely give numerically identical
outputs on most random Pauli inputs, the median makes sense. If they
shouldn't, something is wrong in Pauli_MPS_after_TDME_output_only.
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


def go(N, t_val, gamma_val, weight=None, n_samples=10):
    print(f"\n=== N={N}, t={t_val}, gamma={gamma_val}, weight={weight} ===")
    T_model = 3
    T_target = 30
    mu = 1.0
    J = 1.0
    gamma = [gamma_val] * N

    r_model = t_val / T_model
    r_target = t_val / T_target

    A_u = tdme.construct_TDME_unitary(N, T_model, r=r_model, mu=mu, J=J)
    A_j = tdme.construct_jump_matrices(N, gamma, r=r_model)
    B_u = tdme.construct_TDME_unitary(N, T_target, r=r_target, mu=mu, J=J)
    B_j = tdme.construct_jump_matrices(N, gamma, r=r_target)

    print("Per-sample [T=3 norm | T=30 norm | distance | 1-overlap | weight]")
    for i in range(n_samples):
        np.random.seed(100 + i); torch.manual_seed(100 + i)
        M_in, w = tdme.random_pauli_MPS(N, weight=weight)
        with torch.no_grad():
            M3, _ = tdme.Pauli_MPS_after_TDME_output_only(
                M_in.copy(), T_model, r=r_model,
                all_unitary=A_u, all_jumping=A_j,
                max_bd=64, max_err=1e-10, truncation=True,
            )
            M30, _ = tdme.Pauli_MPS_after_TDME_output_only(
                M_in.copy(), T_target, r=r_target,
                all_unitary=B_u, all_jumping=B_j,
                max_bd=64, max_err=1e-10, truncation=True,
            )
        M3c = M3.astype("complex128")
        M30c = M30.astype("complex128")
        n3 = M3c.norm()
        n30 = M30c.norm()
        d2 = tdme.tensor_network_distance(M3c, M30c).item()
        ovl = abs((M3c.H @ M30c)) / (n3 * n30 + 1e-30)
        print(f"  i={i}  |T=3|={n3:.4e}  |T=30|={n30:.4e}  ||T=3-T=30||^2={d2:.4e}  1-ovl={1-ovl:.4e}  w={w}")


if __name__ == "__main__":
    # ── N=30 reproduction ──
    # Exactly the regime the saved jobs ran: N=30, t=0.8, γ=0.1, T=3 vs T=30.
    # Saved files claim median ~1e-13 — let's verify directly.
    go(N=30, t_val=0.8, gamma_val=0.1, weight=None, n_samples=6)
