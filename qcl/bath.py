"""System-bath teacher: 2N-site (data+bath) evolution, partial trace, data gen."""
import numpy as np
import torch
import torch.nn as nn
import quimb.tensor as qtn

from .pauli import (_EYE4, _PAULI_BASIS, _TRACE_VEC_PAULI, Haar_random_unitary,
                    unitary_to_transfer_matrix_two_site_truncated)
from .states import (Identity_init, operator_assignment_single_site,
                     random_pauli_MPS, Pauli_MPS_weight_1)
from .evolve import evol_two_site



# ──────────────────────────────────────────────────────────────────────
# System-bath extension: helpers for a 2N-site teacher whose layout is
#   d b d b d b ...   (data at even MPS sites, bath at odd MPS sites)
# The teacher has no noise; each of its L unitary layers applies
#   (i) a Haar random 2-qubit unitary between nearest data qubits, embedded
#       as a 3-site Pauli-basis gate U_H ⊗ I_bath on (2i, 2i+1, 2i+2), and
#   (ii) a weak interaction exp(-i g H_db) on every (data, bath) pair.
# Bath is partial-traced at the end (contract the I-component of each bath
# phys leg) before comparing to the student QMLM.
# ──────────────────────────────────────────────────────────────────────

def weak_data_bath_unitary(g, J_b=1.0):
    """Build the 4x4 data-bath weak-interaction unitary exp(-i g (-J_b) (XX + YY)).

    Mirrors the hopping-style H used in TDME_two_site_hamiltonian. `g` is a small
    dimensionless coupling strength (g ~ 0.05). Returns a complex128 torch tensor.
    """
    from scipy.linalg import expm
    X = _PAULI_BASIS[:, :, 1].numpy()
    Y = _PAULI_BASIS[:, :, 2].numpy()
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    H = -J_b * (XX + YY)
    U = expm(-1j * float(g) * H)
    return torch.from_numpy(U).to(torch.complex128)



def three_site_haar_transfer_matrix(U_H):
    """Pauli transfer tensor for the 3-site gate `U_H ⊗ I_bath`.

    U_H is a 4x4 unitary acting on (d_i, d_{i+1}); the middle bath qubit is
    untouched. Returns a tensor of shape (4, 4, 4, 4, 4, 4) indexed as
    (out_d0, out_bath, out_d1, in_d0, in_bath, in_d1), suitable for
    `evol_three_site`.
    """
    U_H = U_H.to(torch.complex128)
    # T_U[out_d0, out_d1, in_d0, in_d1] from the 16x16 two-site transfer matrix
    T_U = unitary_to_transfer_matrix_two_site_truncated(U_H).reshape(4, 4, 4, 4)
    # T3[a=out_d0, b=out_bath, c=out_d1, d=in_d0, e=in_bath, f=in_d1]
    #   = T_U[a, c, d, f] * I4[b, e]
    T3 = torch.einsum('acdf, be -> abcdef', T_U, _EYE4)
    return T3



def evol_three_site(M, T3, site, truncation=False, max_bd=1024, max_err=1E-10):
    """Apply a 3-site Pauli-transfer tensor `T3` to MPS `M` starting at `site`.

    T3 shape: (4,4,4, 4,4,4) with index order (out0, out1, out2, in0, in1, in2).
    The MPS convention follows Identity_init: M[site].data has shape (L, phys, R)
    at each site. QR splits are used (no per-gate truncation; truncation happens
    at the outer per-layer qtn.tensor_network_1d_compress call).
    """
    assert site <= M.L - 3, f"site {site} + 2 exceeds MPS length {M.L}"
    A1 = M[site].data
    A2 = M[site + 1].data
    A3 = M[site + 2].data
    d = A1.shape[1]  # = 4
    DL = A1.shape[0]
    DR = A3.shape[2]

    # Cast to a common complex dtype to avoid mixed-dtype einsum crashes.
    target_dtype = T3.dtype if T3.is_complex() else A1.dtype
    if not target_dtype.is_complex:
        target_dtype = torch.complex128
    if A1.dtype != target_dtype:
        A1 = A1.to(target_dtype)
        A2 = A2.to(target_dtype)
        A3 = A3.to(target_dtype)
    if T3.dtype != target_dtype:
        T3 = T3.to(target_dtype)

    # Contract three site tensors with the gate. Indices:
    #   A1[i, a, b] * A2[b, c, d] * A3[d, e, f] * T3[p, q, r, a, c, e] -> [i, p, q, r, f]
    A_evol = torch.einsum('iab, bcd, def, pqrace -> ipqrf', A1, A2, A3, T3)
    # Shape: (DL, d, d, d, DR)

    # Split back into three tensors via two QRs.
    mat1 = A_evol.reshape(DL * d, d * d * DR)
    Q1, R1 = torch.linalg.qr(mat1)
    new_Dm1 = R1.shape[0]
    new_A1 = Q1.reshape(DL, d, new_Dm1)

    mat2 = R1.reshape(new_Dm1 * d, d * DR)
    Q2, R2 = torch.linalg.qr(mat2)
    new_Dm2 = R2.shape[0]
    new_A2 = Q2.reshape(new_Dm1, d, new_Dm2)
    new_A3 = R2.reshape(new_Dm2, d, DR)

    M[site].modify(data=new_A1)
    M[site + 1].modify(data=new_A2)
    M[site + 2].modify(data=new_A3)
    return 1, M



def three_site_haar_layer(M, T3_list, eo="o", truncation=False, max_bd=1024, max_err=1E-10):
    """Brickwall of 3-site Haar gates on the data sublattice of a 2N-site MPS.

    For a 2N-site MPS with data at even MPS positions:
      eo="o": gates at positions (0,1,2), (4,5,6), (8,9,10), ...
              covers data pairs (d_0,d_1), (d_2,d_3), ...
      eo="e": gates at (2,3,4), (6,7,8), ...
              covers (d_1,d_2), (d_3,d_4), ...

    `T3_list` is a list of 3-site transfer tensors (one per gate position).
    """
    M.astype_('complex128')
    N_total = M.L
    site0 = 0 if eo == "o" else 2
    # Last valid starting site is N_total-3 (gate spans 3 sites). Step is 4.
    Uind = 0
    for site in range(site0, N_total - 2, 4):
        T3 = T3_list[Uind]
        _, M = evol_three_site(M, T3, site, truncation=truncation,
                               max_bd=max_bd, max_err=max_err)
        Uind += 1
    return 1, M



def weak_db_layer(M, T_weak_list, truncation=False, max_bd=1024, max_err=1E-10):
    """Apply 2-site weak gates on every (d_i, b_i) pair = MPS sites (2i, 2i+1).

    `T_weak_list` is a list of 16x16 Pauli transfer matrices (one per pair).
    Uses the existing evol_two_site with a precomputed TM.
    """
    N_total = M.L
    N_data = N_total // 2
    for i in range(N_data):
        site = 2 * i
        tm = T_weak_list[i]
        # evol_two_site uses tm when provided; the U argument is ignored.
        _, M = evol_two_site(M, None, site, "l", truncation=truncation,
                             max_bd=max_bd, max_err=max_err, tm=tm)
    return 1, M



def _compress_mps_with_bath(M, max_bd, max_err, layer):
    """Layer-end compression for the 2N-site MPS. Mirrors the compression logic
    in Pauli_MPS_after_QMLM_output_only but handles the case where the MPS
    carries a size-1 boundary leg inherited from a previous layer (which would
    otherwise double-add a boundary axis each iteration).
    """
    # Pre-compress: swap axes so phys is trailing (lrp layout expected by compress).
    for j_inner, tensor in enumerate(M):
        inds = list(M[j_inner].inds)
        if len(inds) >= 3:
            inds[1], inds[2] = inds[2], inds[1]
            M[j_inner].transpose_(*inds)

    M = qtn.tensor_network_1d_compress(
        M, method="direct", max_bond=max_bd, cutoff=max_err,
        cutoff_mode="rsum2", permute_arrays=False,
    )

    # Re-introduce boundary bonds. After compress, boundary tensors may be rank-2
    # (compress stripped a size-1 bond — first layer case) or rank-3 (a named
    # boundary leg from a previous layer survived compress). Handle both.
    for j in (0, M.L - 1):
        t = M[j]
        name = f"Bath_Boundary_L{layer}_S{j}"
        if t.data.ndim == 2:
            if j == 0:
                new_data = t.data.reshape(1, t.data.shape[0], t.data.shape[1])
                M[j].modify(data=new_data, inds=(name, t.inds[0], t.inds[1]))
            else:
                new_data = t.data.reshape(t.data.shape[0], 1, t.data.shape[1])
                M[j].modify(data=new_data, inds=(t.inds[0], name, t.inds[1]))
        else:
            # Already has a size-1 dangling index from a prior layer; find it and rename.
            shared = set()
            if j > 0:
                shared |= set(t.inds) & set(M[j - 1].inds)
            if j < M.L - 1:
                shared |= set(t.inds) & set(M[j + 1].inds)
            dangling = [
                idx for idx in t.inds
                if idx not in shared
                and not idx.startswith('input')
                and not idx.startswith('k')
            ]
            if dangling:
                M[j].reindex_({dangling[0]: name})

    # Swap back: phys returns to axis 1 (lpr layout).
    for j_inner, tensor in enumerate(M):
        inds = list(M[j_inner].inds)
        if len(inds) >= 3:
            inds[1], inds[2] = inds[2], inds[1]
            M[j_inner].transpose_(*inds)
    return M



def partial_trace_bath(M_2N, N):
    """Trace out odd (bath) sites of a 2N-site Pauli-basis MPS.

    Contracts each bath site's physical leg with _TRACE_VEC_PAULI = [1,0,0,0]
    (picks the Identity component), absorbs the resulting (DL, DR) matrix into
    the adjacent data tensor's bond, and builds an N-site MPS on the data qubits.

    The returned MPS has physical indices 'input{j}' for j=0..N-1, matching the
    Identity_init / Pauli_MPS_weight_1 convention. Caller is responsible for
    reindexing to 'k{j}' before comparing to a student output via
    tensor_network_distance.
    """
    N_total = 2 * N
    assert M_2N.L == N_total, f"Expected MPS of length {N_total}, got {M_2N.L}"

    # Extract (L, 4, R) arrays. Trust the Identity_init convention throughout
    # Pauli_MPS_after_QMLM_output_only. Boundary tensors may be rank-2 (no size-1
    # boundary bond) — unsqueeze to restore the standard rank-3 shape.
    arrays = []
    for i in range(N_total):
        A = M_2N[i].data
        if A.ndim == 2:
            if i == 0:
                A = A.unsqueeze(0)          # (1, 4, R)
            elif i == N_total - 1:
                A = A.unsqueeze(-1)         # (L, 4, 1)
        arrays.append(A)

    data_arrays = [arrays[2 * i].clone() for i in range(N)]

    trace_vec = _TRACE_VEC_PAULI
    for i in range(N):
        bath_A = arrays[2 * i + 1]
        tv = trace_vec.to(bath_A.dtype)
        bath_mat = torch.einsum('lpr, p -> lr', bath_A, tv)  # (DL_b, DR_b)

        if i < N - 1:
            # Absorb into the left bond of data-(i+1).
            nxt = data_arrays[i + 1]
            if bath_mat.dtype != nxt.dtype:
                bath_mat = bath_mat.to(nxt.dtype)
            data_arrays[i + 1] = torch.einsum('Ll, lpr -> Lpr', bath_mat, nxt)
        else:
            # Last bath: absorb into the right bond of data-(N-1).
            cur = data_arrays[i]
            if bath_mat.dtype != cur.dtype:
                bath_mat = bath_mat.to(cur.dtype)
            data_arrays[i] = torch.einsum('lpr, rR -> lpR', cur, bath_mat)

    data_arrays = [t.to(torch.complex128) for t in data_arrays]

    M_data = qtn.MatrixProductState(data_arrays)
    # Match Identity_init index convention: phys leg named 'input{j}'.
    for j in range(N):
        indices = M_data[j].inds
        M_data[j].reindex({indices[1]: indices[2], indices[2]: indices[1]}, inplace=True)
        M_data[j].reindex_({f'k{j}': f'input{j}'})
    return M_data



def Pauli_MPS_weight_1_with_bath(N):
    """Generate 3N weight-1 Pauli MPS on the 2N-site (data + bath) layout.

    Each MPS has data sites (even MPS positions) initialized to [1,0,0,0] except
    one data site set to X/Y/Z; bath sites (odd positions) are always [1,0,0,0]
    (Identity in Pauli basis). Ordering matches Pauli_MPS_weight_1(N) so that the
    i-th entry corresponds to the same (data_site, Pauli) pair.
    """
    N_total = 2 * N
    pauli_ops = ['X', 'Y', 'Z']
    MPS_list = []
    for data_site in range(N):
        mps_site = 2 * data_site
        for op in pauli_ops:
            M = Identity_init(N_total, bond_dim=1, phys_dim=4)
            M = operator_assignment_single_site(M, mps_site, op)
            MPS_list.append(M)
    return MPS_list



class QMLM_with_bath_output_only(nn.Module):
    """Non-trainable teacher for the system-bath learning setup.

    The teacher acts on a 2N-site MPS laid out as `d b d b d b ...` (even MPS
    positions = data, odd = bath). Each of its L layers applies:
      1. Haar random 2-qubit unitaries between nearest data qubits, embedded
         as 3-site Pauli-basis gates U_H ⊗ I_bath (brickwall: odd, then even
         data pairs).
      2. Weak 2-qubit interactions exp(-i g (-J_b)(XX+YY)) on every (data,bath)
         pair.
    No noise. The output 2N-site MPS is partial-traced over the bath qubits by
    the caller (see `partial_trace_bath`) before being used as a training target.
    """
    def __init__(self, N, L, haar_list=None, weak_list=None,
                 coupling_strength=0.05, J_b=1.0,
                 max_bd=64, max_err=1E-10):
        super().__init__()
        self.N_data = int(N)
        self.N_total = 2 * int(N)
        self.L = int(L)
        self.max_bd = max_bd
        self.max_err = max_err
        self.coupling_strength = float(coupling_strength)
        self.J_b = float(J_b)

        N_pairs_o = N // 2
        N_pairs_e = (N - 1) // 2

        if haar_list is None:
            haar_list = []
            for _ in range(self.L):
                layer_haars = [Haar_random_unitary(4) for _ in range(N_pairs_o + N_pairs_e)]
                haar_list.append(layer_haars)
        else:
            assert len(haar_list) == self.L, f"haar_list must have {self.L} layers"
            for l in range(self.L):
                assert len(haar_list[l]) == N_pairs_o + N_pairs_e, \
                    f"haar_list[{l}] must have {N_pairs_o + N_pairs_e} gates"
        self.haar_list = haar_list

        if weak_list is None:
            U_weak = weak_data_bath_unitary(coupling_strength, J_b=J_b)
            weak_list = [[U_weak for _ in range(N)] for _ in range(self.L)]
        else:
            assert len(weak_list) == self.L
            for l in range(self.L):
                assert len(weak_list[l]) == N
        self.weak_list = weak_list

        # Precompute all Pauli transfer matrices once. Non-trainable; stored as
        # plain Python lists of complex128 torch tensors (nn.Parameter would
        # require them as tensor leaves of known shape, which is awkward for
        # nested lists).
        self._T3_odd = []
        self._T3_even = []
        self._T_weak = []
        for l in range(self.L):
            T3_o = [three_site_haar_transfer_matrix(haar_list[l][p]).to(torch.complex128)
                    for p in range(N_pairs_o)]
            T3_e = [three_site_haar_transfer_matrix(haar_list[l][N_pairs_o + p]).to(torch.complex128)
                    for p in range(N_pairs_e)]
            T_w = [unitary_to_transfer_matrix_two_site_truncated(
                        weak_list[l][p].to(torch.complex128)
                   ).to(torch.complex128)
                   for p in range(N)]
            self._T3_odd.append(T3_o)
            self._T3_even.append(T3_e)
            self._T_weak.append(T_w)

    def forward(self, M_input_2N, truncation=True):
        """Evolve a 2N-site Pauli-basis MPS through L layers of the bath teacher.

        With `truncation=True`, compression is applied after every sub-layer
        (Haar-odd, Haar-even, weak-db) rather than only once per outer layer.
        Intermediate bond dims from 3-site gates can reach d^3=64 within a
        single sub-layer; per-sub-layer compression keeps them capped at
        `self.max_bd` throughout the forward pass.
        """
        assert M_input_2N.L == self.N_total, \
            f"Input MPS must have length {self.N_total}, got {M_input_2N.L}"
        M = M_input_2N
        M.astype_('complex128')
        sub = 0
        for l in range(self.L):
            _, M = three_site_haar_layer(
                M, self._T3_odd[l], eo="o",
                truncation=False, max_bd=self.max_bd, max_err=self.max_err,
            )
            if truncation:
                M = _compress_mps_with_bath(M, self.max_bd, self.max_err, layer=sub)
                sub += 1
            _, M = three_site_haar_layer(
                M, self._T3_even[l], eo="e",
                truncation=False, max_bd=self.max_bd, max_err=self.max_err,
            )
            if truncation:
                M = _compress_mps_with_bath(M, self.max_bd, self.max_err, layer=sub)
                sub += 1
            _, M = weak_db_layer(
                M, self._T_weak[l],
                truncation=False, max_bd=self.max_bd, max_err=self.max_err,
            )
            if truncation:
                M = _compress_mps_with_bath(M, self.max_bd, self.max_err, layer=sub)
                sub += 1
        return M



def get_input_and_output_MPS_with_bath(
    N, L,
    haar_list=None, weak_list=None,
    coupling_strength=0.05, J_b=1.0,
    truncation=True, max_bd=64, max_err=1E-10,
    num_threads=None,
):
    """Generate weight-1 data inputs and bath-traced targets for the QMLM-with-bath setup.

    Builds:
      - `MPS_weight1_data`: 3N weight-1 Pauli MPS on N data qubits (fed to the student QMLM).
      - `target_mps_list`:  3N N-site MPS obtained by evolving the corresponding 2N-site
                            embedded input through QMLM_with_bath_output_only and tracing
                            out the bath. Phys legs are already reindexed to 'k{i}'.

    Returns (MPS_weight1_data, target_mps_list, haar_list, weak_list).
    """
    # Instantiate the teacher once. The teacher generates random Haar unitaries
    # internally if `haar_list` is None, so we read them back afterwards so callers
    # can persist them (mirroring how `param` is returned from the standard helper).
    teacher = QMLM_with_bath_output_only(
        N, L, haar_list=haar_list, weak_list=weak_list,
        coupling_strength=coupling_strength, J_b=J_b,
        max_bd=max_bd, max_err=max_err,
    )
    haar_list = teacher.haar_list
    weak_list = teacher.weak_list

    MPS_weight1_data = Pauli_MPS_weight_1(N)
    MPS_weight1_with_bath = Pauli_MPS_weight_1_with_bath(N)
    total = len(MPS_weight1_data)

    if num_threads is None:
        num_threads = 1

    target_mps_list = []
    with torch.no_grad():
        for counter, inp_2N in enumerate(MPS_weight1_with_bath):
            M_out_2N = teacher.forward(inp_2N.copy(), truncation=truncation)
            M_target = partial_trace_bath(M_out_2N, N)
            # Reindex 'input{i}' -> 'k{i}' to match the target-MPS convention used
            # throughout the training loops.
            for i, tensor in enumerate(M_target):
                tensor.reindex_({f'input{i}': f'k{i}'})
            target_mps_list.append(M_target)
    print(f"N{N}, L{L}, g{coupling_strength}: processed {total} with-bath MPS", flush=True)

    return MPS_weight1_data, target_mps_list, haar_list, weak_list



def get_random_input_output_MPS_with_bath(
    N, L,
    no_sample=30,
    haar_list=None, weak_list=None,
    coupling_strength=0.05, J_b=1.0,
    truncation=True, max_bd=64, max_err=1E-10,
):
    """Random-Pauli variant of `get_input_and_output_MPS_with_bath` used for testing.

    Generates `no_sample` random Pauli MPS on N data qubits, embeds each into the
    2N-site layout (bath=Identity), evolves through the teacher, partial-traces,
    and returns the data-side inputs + targets. No parallelism — this is called
    sequentially in the testing loop.
    """
    teacher = QMLM_with_bath_output_only(
        N, L, haar_list=haar_list, weak_list=weak_list,
        coupling_strength=coupling_strength, J_b=J_b,
        max_bd=max_bd, max_err=max_err,
    )
    haar_list = teacher.haar_list
    weak_list = teacher.weak_list

    N_total = 2 * N
    mps_data_list = []
    mps_2N_list = []
    for _ in range(no_sample):
        M_data, _ = random_pauli_MPS(N)
        # Build the matching 2N-site input: same Pauli string on data sites, bath = Identity.
        M_2N = Identity_init(N_total, bond_dim=1, phys_dim=4)
        for data_site in range(N):
            mps_site = 2 * data_site
            # Read the data-site operator from M_data by inspecting its phys coeff.
            phys_ind = f'input{data_site}'
            if phys_ind not in M_data[data_site].inds:
                phys_ind = f'k{data_site}'
            # Extract the phys vector (bond dims are 1 by construction of weight-1/random Pauli MPS).
            vec = M_data[data_site].data.reshape(-1)
            # operator_assignment_single_site only accepts X/Y/Z; detect which (or Identity).
            if torch.allclose(vec.to(torch.float64), torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)):
                continue  # identity — leave the bath-embedded MPS untouched at this site.
            if torch.allclose(vec.to(torch.float64), torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)):
                M_2N = operator_assignment_single_site(M_2N, mps_site, "X")
            elif torch.allclose(vec.to(torch.float64), torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)):
                M_2N = operator_assignment_single_site(M_2N, mps_site, "Y")
            elif torch.allclose(vec.to(torch.float64), torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)):
                M_2N = operator_assignment_single_site(M_2N, mps_site, "Z")
        mps_data_list.append(M_data)
        mps_2N_list.append(M_2N)

    target_mps_list = []
    with torch.no_grad():
        for inp_2N in mps_2N_list:
            M_out_2N = teacher.forward(inp_2N.copy(), truncation=truncation)
            M_target = partial_trace_bath(M_out_2N, N)
            for i, tensor in enumerate(M_target):
                tensor.reindex_({f'input{i}': f'k{i}'})
            target_mps_list.append(M_target)

    return mps_data_list, target_mps_list, haar_list, weak_list

