"""Consistency check at N=6, T=L=3, g=0.

Compare two MPSs element-wise via dense contraction:
  - `student`: Haar brickwall on N data qubits, no bath, no noise.
  - `teacher`: bath extension with matched Haar gates, g=0, partial-traced.

At g=0 the bath does nothing, so both MPSs must be identical as operators.

truncation=False here so there's no compression error to mask a bug.
"""
import os, sys, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Learning_Lindbladian"))
import numpy as np, torch
np.random.seed(11); torch.manual_seed(11)

import TDME_Trott as tdme

N, L = 6, 3
g = 0.0

t0 = time.time()
teacher = tdme.QMLM_with_bath_output_only(
    N, L, coupling_strength=g, J_b=1.0, max_bd=64, max_err=1e-12,
)
haar_list = teacher.haar_list
T_U_list = [
    [tdme.unitary_to_transfer_matrix_two_site_truncated(H.to(torch.complex128)).to(torch.complex128)
     for H in haar_list[l]] for l in range(L)
]
N_pairs_o, N_pairs_e = N // 2, (N - 1) // 2

def student_forward(M_in):
    M = M_in.copy(); M.astype_('complex128')
    for l in range(L):
        for p in range(N_pairs_o):
            _, M = tdme.evol_two_site(M, None, 2*p, "l", truncation=False, tm=T_U_list[l][p])
        for p in range(N_pairs_e):
            _, M = tdme.evol_two_site(M, None, 2*p+1, "l", truncation=False, tm=T_U_list[l][N_pairs_o+p])
    return M

def mps_to_dense(M, N_phys):
    inds = [f'k{j}' for j in range(N_phys)]
    return np.array(M.contract(output_inds=inds).data)

data_in = tdme.Pauli_MPS_weight_1(N)
bath_in = tdme.Pauli_MPS_weight_1_with_bath(N)

max_diff = 0.0
for i in range(len(data_in)):
    ti = time.time()
    M_s = student_forward(data_in[i])
    for j, t in enumerate(M_s): t.reindex_({f'input{j}': f'k{j}'})
    M_t_full = teacher.forward(bath_in[i].copy(), truncation=False)
    M_t = tdme.partial_trace_bath(M_t_full, N)
    for j, t in enumerate(M_t): t.reindex_({f'input{j}': f'k{j}'})
    A = mps_to_dense(M_s, N)
    B = mps_to_dense(M_t, N)
    diff = float(np.max(np.abs(A - B)))
    max_diff = max(max_diff, diff)
    print(f"  input {i:2d}: |A - B|_inf = {diff:.3e}  ({time.time()-ti:.1f}s)", flush=True)

print(f"\nN={N}, L=T={L}, g=0: MAX |student - traced_teacher|_inf = {max_diff:.3e}  ({time.time()-t0:.1f}s total)")
if max_diff < 1e-10:
    print("PASS")
else:
    print("FAIL")
    sys.exit(1)
