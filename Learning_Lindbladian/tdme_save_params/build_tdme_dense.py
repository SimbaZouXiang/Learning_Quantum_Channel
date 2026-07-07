"""Build the full TDME Lindbladian channel as a 4^N x 4^N dense Pauli transfer matrix.

The TDME 2nd-order Trotter layer applies, in order:
  1) one-site jump PTMs on every site (J_i = diag(1, beta, beta, 1) with beta=exp(-gamma_i r/4))
  2) two-site odd brickwall (each U is 4x4; PTM is 16x16 = (4,4,4,4))   — half step
  3) two-site even brickwall                                            — full step
  4) two-site odd brickwall                                             — half step
  5) one-site jump PTMs again

After L such layers we materialise the full 4^N x 4^N matrix.

M_full has shape (d, d) where d=4^N. We reshape to (4,4,...,4, d) — N "output"
legs followed by one composite "input" leg — and apply local PTMs via tensordot.
"""
import sys, os
import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import torch
import TDME_Trott as tdme


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _apply_one_site_layer(M, ptm_list, N, d_loc=4):
    """Apply a 1-site PTM at each site i to axis i of M (which is reshaped to (d_loc,)*N + (d_total,))."""
    for i in range(N):
        T = _to_np(ptm_list[i]).astype(np.float64)
        # contract: M_new[..., a_i, ...] = sum_c T[a_i, c] M[..., c, ...]
        M = np.tensordot(T, M, axes=([1], [i]))
        # new axis 0 is a_i; move it back to position i
        M = np.moveaxis(M, 0, i)
    return M


def _apply_two_site_brickwall(M, U_list, eo, N, d_loc=4):
    """Apply a brickwall of 2-site unitaries (each U a 4x4 complex matrix).
    For each pair, convert U -> 16x16 PTM, reshape to (4,4,4,4), then contract.
    """
    if eo == "o":
        sites = list(range(0, N - 1, 2))
    else:
        sites = list(range(1, N - 1, 2))
    # construct_TDME_unitary over-allocates one extra unitary at the boundary
    # for both odd and even halves; two_site_layer silently consumes only the
    # first len(sites). Mirror that behavior here.
    if len(U_list) < len(sites):
        raise ValueError(f"eo={eo}: expected at least {len(sites)} unitaries, got {len(U_list)}")
    for site, U in zip(sites, U_list[:len(sites)]):
        U_np = _to_np(U)
        # PTM = (1/4) Tr[P^a U P^c U^dagger] etc. — use existing helper, then reshape
        T16 = _to_np(tdme.unitary_to_transfer_matrix_two_site(torch.as_tensor(U_np))).astype(np.float64)
        T = T16.reshape(d_loc, d_loc, d_loc, d_loc)  # (out_i, out_i+1, in_i, in_i+1)
        # M_new[..., a_site, a_site+1, ...] = sum_{c,d} T[a, b, c, d] M[..., c, d, ...]
        M = np.tensordot(T, M, axes=([2, 3], [site, site + 1]))
        # current first two axes are (out_i, out_i+1); move them back to (site, site+1)
        M = np.moveaxis(M, [0, 1], [site, site + 1])
    return M


def build_tdme_dense(N, L, mu, gamma, J=1.0, t=1.0):
    """Return the full Pauli transfer matrix of the TDME 2nd-order Trotter channel
    with N qubits and L layers as a 4^N x 4^N float64 numpy array.

    Parameters
    ----------
    gamma : float or sequence of length N. Per-site dephasing-like rates used in
        the jump matrices J_i = diag(1, exp(-gamma_i r/4), exp(-gamma_i r/4), 1).
    """
    if np.isscalar(gamma):
        gamma_vec = [float(gamma)] * N
    else:
        gamma_vec = list(gamma)

    r = t / L
    all_unitary = tdme.construct_TDME_unitary(N, L, r=r, mu=mu, J=J)
    all_jumping = tdme.construct_jump_matrices(N, gamma_vec, r=r)

    d_loc = 4
    d_total = d_loc ** N

    # Start with identity channel as (d_total, d_total)
    M = np.eye(d_total, dtype=np.float64)
    # Reshape so the N output legs live at axes 0..N-1 and the composite input
    # leg is the last axis. Standard "matrix as state vector" trick.
    M = M.reshape((d_loc,) * N + (d_total,))

    for layer in range(L):
        # 1) jumps
        M = _apply_one_site_layer(M, all_jumping, N)
        # 2) odd brickwall (half step)
        M = _apply_two_site_brickwall(M, all_unitary[layer][:N // 2], "o", N)
        # 3) even brickwall (full step)
        M = _apply_two_site_brickwall(M, all_unitary[layer][N // 2:], "e", N)
        # 4) odd brickwall again (half step)
        M = _apply_two_site_brickwall(M, all_unitary[layer][:N // 2], "o", N)
        # 5) jumps
        M = _apply_one_site_layer(M, all_jumping, N)

    return M.reshape(d_total, d_total)


# ── validation: compare against column-by-column application of the
# existing Pauli_MPS_after_TDME_output_only on small N ────────────────

def _pauli_basis_mps(N, idx):
    """Build an MPS for the idx-th Pauli basis state (idx in 0..4^N-1).
    Index decoded base-4, LSD = site 0 (matches the (4,)*N reshape order in
    build_tdme_dense).  Uses TDME_Trott.Identity_init so every tensor is 3D
    (left_bond, phys, right_bond) — what the teacher's evol_one_site expects.
    """
    M = tdme.Identity_init(N, bond_dim=1, phys_dim=4)
    op_name = {0: None, 1: "X", 2: "Y", 3: "Z"}
    sites_changed, ops = [], []
    # site 0 is MOST significant (matches quimb's to_dense leg ordering and the
    # (4,)*N reshape used inside build_tdme_dense, both default to C order).
    for site in range(N):
        digit = (idx >> (2 * (N - 1 - site))) & 3
        if digit != 0:
            sites_changed.append(site)
            ops.append(op_name[digit])
    if sites_changed:
        M = tdme.operator_assignment(M, sites_changed, ops)
    return M


def validate_at_N4():
    """Cross-check dense PTM against teacher's MPS-application on a few inputs."""
    import quimb.tensor as qtn
    N = 4
    L = 3
    mu, J_ = 1.0, 1.0
    t_ = 1.0
    gamma_ = 0.1
    M_dense = build_tdme_dense(N, L, mu, gamma_, J=J_, t=t_)
    print(f"  built dense ({M_dense.shape})", flush=True)

    r = t_ / L
    all_unitary = tdme.construct_TDME_unitary(N, L, r=r, mu=mu, J=J_)
    all_jumping = tdme.construct_jump_matrices(N, [gamma_] * N, r=r)

    # Check a few random Pauli basis indices.
    d_total = 4 ** N
    rng = np.random.default_rng(0)
    test_idxs = rng.choice(d_total, size=8, replace=False)
    max_err = 0.0
    for idx in test_idxs:
        mps_in = _pauli_basis_mps(N, int(idx))
        # ensure data is complex128 torch (teacher cast path uses .to(other.dtype))
        for t in mps_in.tensors:
            t.modify(data=t.data.to(torch.complex128) if torch.is_tensor(t.data)
                            else torch.from_numpy(np.asarray(t.data)).to(torch.complex128))
        mps_out, _ = tdme.Pauli_MPS_after_TDME_output_only(
            mps_in.copy(), L, r=r, all_unitary=all_unitary, all_jumping=all_jumping,
            max_bd=512, max_err=1e-12, truncation=False,
        )
        out_vec = mps_out.to_dense([f"input{i}" for i in range(N)])
        out_vec = np.asarray(out_vec).reshape(-1)
        # compare against dense column
        col = M_dense[:, idx]
        err = float(np.max(np.abs(out_vec - col)))
        max_err = max(max_err, err)
        print(f"    idx={idx:>4d}  ||MPS_out - M_dense[:, idx]||_inf = {err:.3e}", flush=True)
    print(f"  worst error across {len(test_idxs)} basis cols: {max_err:.3e}", flush=True)
    return max_err


if __name__ == "__main__":
    print("=== validate_at_N4 ===", flush=True)
    err = validate_at_N4()
    if err < 1e-8:
        print("PASS", flush=True)
    else:
        print(f"FAIL (max err {err:.3e})", flush=True)
        sys.exit(1)
