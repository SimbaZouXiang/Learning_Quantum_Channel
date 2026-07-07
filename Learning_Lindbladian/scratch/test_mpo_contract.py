"""Smoke test: compute the Frobenius distance between two QMLM channels
two different ways and check they agree.

Method 1: Direct MPO-MPO contraction. Compute Tr(A^† B), Tr(A^† A), Tr(B^† B)
         via tensor-network contraction, then ||A-B||^2 = Tr(A^†A)+Tr(B^†B)-2Re Tr(A^†B).

Method 2: For small N, enumerate the full Pauli basis (4^N inputs) and sum
         ||A(σ_α) - B(σ_α)||^2; divide by 2^N to recover the PTM Frobenius.

For N=4 these should match to machine precision.
"""
import os, sys, warnings
warnings.filterwarnings("ignore", message="Casting complex")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import itertools
import numpy as np
import torch
torch.set_num_threads(1)
import quimb.tensor as qtn
import TDME_Trott as tdme

N = 4
T = 2


def _rename_internal(tn, suffix):
    """Append `suffix` to every internal-bond index (those not in input{i}/k{i})."""
    physical = set(f'input{i}' for i in range(N)) | set(f'k{i}' for i in range(N))
    rename_map = {}
    for t in tn.tensors:
        for ind in t.inds:
            if ind not in physical and ind not in rename_map:
                rename_map[ind] = f"{ind}_{suffix}"
    return tn.reindex(rename_map)


def _to_plain(tn):
    """Convert any PTensor in tn into a plain Tensor with uniform complex128
    dtype, so quimb/cotengra can contract without dtype-mismatch errors.
    """
    new_tensors = []
    for t in tn.tensors:
        plain = t.unparametrize() if hasattr(t, "unparametrize") else t.copy()
        # Force complex128 so all tensors share a dtype.
        if torch.is_tensor(plain.data):
            plain.modify(data=plain.data.to(torch.complex128))
        else:
            plain.modify(data=plain.data.astype(np.complex128))
        new_tensors.append(plain)
    return qtn.TensorNetwork(new_tensors)


def hs_inner(A, B):
    """Compute Tr(A^† B) for two MPO tensor networks in Pauli basis.

    A and B have physical indices 'input{i}' (column α) and 'k{i}' (row β).
    Result = sum_{α,β} A[β,α]^* B[β,α].
    """
    A_plain = _to_plain(A)
    B_plain = _to_plain(B)
    for t in A_plain.tensors:
        t.modify(data=np.conj(t.data) if not torch.is_tensor(t.data) else t.data.conj())
    A_renamed = _rename_internal(A_plain, "A")
    B_renamed = _rename_internal(B_plain, "B")
    combined = A_renamed & B_renamed
    return combined.contract().item()


def frobenius_sq_via_contraction(A, B):
    """||A - B||_F^2 = ⟨A,A⟩ + ⟨B,B⟩ - 2 Re ⟨A,B⟩."""
    AA = hs_inner(A, A)
    BB = hs_inner(B, B)
    AB = hs_inner(A, B)
    return float(np.real(AA + BB - 2 * AB))


def frobenius_sq_via_enumeration(A, B):
    """Sum_{α over the complete 4^N Pauli basis} ||A(σ_α) - B(σ_α)||²_{MPS-coeff}.

    For an MPO that stores Pauli-basis coefficients c_β with σ_β unnormalised,
    this sum equals ||M_A - M_B||²_F  (the Frobenius distance of the
    Pauli-transfer matrices) — no 2^N factor needed.
    """
    pauli_ops = ['I', 'X', 'Y', 'Z']
    total = 0.0
    for choice in itertools.product(pauli_ops, repeat=N):
        M = tdme.Identity_init(N, bond_dim=1, phys_dim=4)
        for site, op in enumerate(choice):
            if op != 'I':
                M = tdme.operator_assignment_single_site(M, site, op)
        out_A = A | M
        out_B = B | M
        total += tdme.tensor_network_distance(
            out_A.astype("complex128"), out_B.astype("complex128"),
        ).item()
    return total


def main():
    # Two random QMLM channels with different params.
    np.random.seed(0); torch.manual_seed(0)
    p_dep = torch.ones(T, N, dtype=torch.float64) * 0.05
    param_A = (torch.rand(T, N, 16, dtype=torch.float64) + 1j*torch.rand(T, N, 16, dtype=torch.float64))
    param_B = (torch.rand(T, N, 16, dtype=torch.float64) + 1j*torch.rand(T, N, 16, dtype=torch.float64))
    model_A = tdme.QMLM(N, T, param=param_A, p_depolar=p_dep)
    model_B = tdme.QMLM(N, T, param=param_B, p_depolar=p_dep)
    A_mpo = model_A.get_MPO(noise_type="depolarizing")
    B_mpo = model_B.get_MPO(noise_type="depolarizing")

    # First: cross-check the self-overlap ||A||_F^2 two ways.
    AA_contract = hs_inner(A_mpo, A_mpo)
    AA_enum = frobenius_sq_via_enumeration(A_mpo, B_mpo)  # placeholder, computed below
    # ||A||² via enumeration = Σ_α ||A(σ_α)||²_MPS = Σ_α A(σ_α).H @ A(σ_α)
    AA_enum = 0.0
    for choice in itertools.product(['I','X','Y','Z'], repeat=N):
        M = tdme.Identity_init(N, bond_dim=1, phys_dim=4)
        for site, op in enumerate(choice):
            if op != 'I':
                M = tdme.operator_assignment_single_site(M, site, op)
        out_A = (A_mpo | M).astype("complex128")
        AA_enum += (out_A.H @ out_A).real
    print(f"  ||A||² via contraction               : {AA_contract.real:.6e}")
    print(f"  ||A||² via enumeration of σ_α        : {AA_enum:.6e}")
    print(f"  ratio                                : {AA_contract.real / AA_enum:.6f}")

    AB_contract = hs_inner(A_mpo, B_mpo)
    BB_contract = hs_inner(B_mpo, B_mpo)
    AB_enum = 0.0 + 0.0j
    BB_enum = 0.0
    dist_enum = 0.0
    for choice in itertools.product(['I','X','Y','Z'], repeat=N):
        M = tdme.Identity_init(N, bond_dim=1, phys_dim=4)
        for site, op in enumerate(choice):
            if op != 'I':
                M = tdme.operator_assignment_single_site(M, site, op)
        out_A = (A_mpo | M).astype("complex128")
        out_B = (B_mpo | M).astype("complex128")
        AB_enum += (out_A.H @ out_B)
        BB_enum += (out_B.H @ out_B).real
        dist_enum += tdme.tensor_network_distance(out_A, out_B).item()
    print(f"  Tr(A†B) via contraction              : {AB_contract:.6e}")
    print(f"  Tr(A†B) via enumeration              : {AB_enum:.6e}")
    print(f"  ||B||²  via contraction              : {BB_contract.real:.6e}")
    print(f"  ||B||²  via enumeration σ_α          : {BB_enum:.6e}")
    print(f"  AA+BB-2Re(AB) from contractions      : {(AA_contract+BB_contract-2*AB_contract).real:.6e}")
    print(f"  AA+BB-2Re(AB) from enumerations      : {AA_enum+BB_enum-2*AB_enum.real:.6e}")
    print(f"  sum_alpha tn_distance(A, B) over σ_α : {dist_enum:.6e}")
    print()

    # Method 1: direct contraction
    d2_contract = frobenius_sq_via_contraction(A_mpo, B_mpo)
    print(f"  ||A-B||² via direct MPO-MPO contraction : {d2_contract:.6e}")

    # Method 2: enumeration (4^N = 256 at N=4)
    d2_enum = frobenius_sq_via_enumeration(A_mpo, B_mpo)
    print(f"  ||A-B||² via Pauli-basis enumeration     : {d2_enum:.6e}")

    rel_diff = abs(d2_contract - d2_enum) / max(abs(d2_contract), 1e-12)
    print(f"  relative difference                       : {rel_diff:.4e}")
    if rel_diff < 1e-6:
        print("  PASS: methods agree to machine precision")
    else:
        print("  FAIL: methods disagree")


if __name__ == "__main__":
    main()
