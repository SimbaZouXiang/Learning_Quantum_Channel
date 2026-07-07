"""Layer-by-layer compressed MPS forward path — O(N·D³) per layer.

Applies each circuit layer directly to a running MPS with QR/SVD bond
truncation instead of building the full circuit MPO.
"""
import numpy as np
import torch
import quimb.tensor as qtn

from .pauli import (_DIR_X, _DIR_Y, _DIR_Z,
                    unitary_to_transfer_matrix_two_site_truncated)
from .noise import (dephasing_noise_transfer_matrix,
                    depolarization_noise_transfer_matrix)


def evol_one_site(M, U, site, dir = "l", truncation = False, max_bd = 1024, max_err=1E-10):
    A = M[site].data
    if not isinstance(U, torch.Tensor):
        U = torch.tensor(U, dtype=A.dtype, device=A.device)
    if A.dtype != U.dtype:
        U = U.to(A.dtype)
    A_evol = torch.einsum('l p r, s p -> l s r', A, U)
    M[site].modify(data = A_evol)

    # Ensure the next site is casted to the same dtype to prevent tensordot mismatched dtype
    if site < M.L - 1 and M[site+1].data.dtype != A_evol.dtype:
        M[site+1].modify(data = M[site+1].data.to(A_evol.dtype))

    # NOTE: every transfer matrix passed in here is diagonal in the Pauli
    # basis (depolarization, dephasingX/Y/Z, jump matrices) so applying it
    # site-by-site does NOT break the MPS canonical form. The previous
    # `M.left_canonicalize()` call here ran a full O(N*D^3) sweep per site
    # — i.e. one_site_layer ended up doing N^2 left-canonicalizations per
    # call (the dominant cost of Pauli_MPS_after_TDME_output_only at
    # N=30, T=30, D=64; ~50 min/sample on a single core). Removing it
    # gives a ~N speedup per layer for free.
    return 1, M



def evol_two_site(M, U, site, dir = "l", truncation = False, max_bd = 1024, max_err=1E-10, tm=None):
    ## Evolve the pure state by U (not necessarily unitary)
    ## Assuming U is two-site
    ## site is the first site # of the unitary (1 to N-1)
    # If a precomputed transfer matrix is supplied, skip the einsum conversion.
    U = tm if tm is not None else unitary_to_transfer_matrix_two_site_truncated(U)
    assert site<=M.L-1
    A1 = M[site].data
    A2 = M[site+1].data
    d1 = A1.shape[1]
    d2 = A2.shape[1]
    DL = A1.shape[0]
    DR = A2.shape[2]
    U = torch.reshape(U,(d1,d2,d1,d2))
    # Promote U up to A's dtype rather than casting A down — the MPS data may
    # be complex128 (cast up before tensor_network_distance), and downcasting
    # complex→real silently drops the imaginary part.
    target_dtype = torch.promote_types(A1.dtype, A2.dtype)
    target_dtype = torch.promote_types(target_dtype, U.dtype)
    if A1.dtype != target_dtype:
        A1 = A1.to(target_dtype)
    if A2.dtype != target_dtype:
        A2 = A2.to(target_dtype)
    if U.dtype != target_dtype:
        U = U.to(target_dtype)
    A_evol = torch.einsum('iab,bcl,jkac->ijkl', A1, A2, U)
    A_evol_mat = A_evol.reshape(DL*d1,DR*d2)

    # Cap intermediate bond dimension at max_bd here, so bond dim doesn't
    # explode through the 3 two-site layers + 1-site layers within a single
    # Trotter step before the per-layer compress. With QR alone, rank can
    # grow as min(DL*d1, DR*d2) per call (e.g. 64 -> 256 -> 1024 across
    # consecutive gates), making the final compress catastrophically slow
    # (SVD of 1000x1000+ matrices). Doing a truncated SVD here keeps the
    # bond dim at <= max_bd throughout the layer.
    if truncation and max_bd is not None and max_bd > 0 and min(A_evol_mat.shape) > max_bd:
        U_, S_, Vh_ = torch.linalg.svd(A_evol_mat, full_matrices=False)
        keep = max_bd
        # Also drop singular values below the cutoff (relative to the largest).
        if max_err is not None and max_err > 0 and S_.numel() > 0:
            thresh = max_err * S_[0]
            keep_by_cutoff = int((S_ > thresh).sum().item())
            keep = max(1, min(keep, keep_by_cutoff))
        U_ = U_[:, :keep]
        S_ = S_[:keep]
        Vh_ = Vh_[:keep, :]
        AL = (U_ * S_).reshape(DL, d1, keep)
        AR = Vh_.reshape(keep, d2, DR)
    else:
        Q, R = torch.linalg.qr(A_evol_mat)
        if(dir == "l"):
            AL = Q.reshape(DL,d1,len(R))
            AR = R.reshape(len(R), d2, DR)

    M[site].modify(data = AL)
    M[site+1].modify(data = AR)
    return 1, M


def two_site_layer(M, Us, eo = "o", truncation = False, max_bd = 1024, max_err = 1E-10, tms=None):
    ## Apply two site operator at each even/odd sites
    ## Note - right canonical form assumed in beginning, and left canonical form in the end
    ## Us = 2-site unitary of length N/2 ("o") or N/2-1 ("e")

    N = M.L
    M.astype('complex128')
    normS = 1.
    Uind = 0
    if(eo == "o"):
        site0 = 0
    else:
        site0 = 1

    for site in range(site0, N-1, 2):
        U = Us[Uind]
        tm = tms[Uind] if tms is not None else None
        norm_site, M = evol_two_site(M, U, site, "l", truncation = truncation, max_bd = max_bd, max_err = max_err, tm=tm)
        Uind += 1

    return normS,M


def one_site_layer(M, Us, truncation = False, max_bd = 1024, max_err = 1E-10):
    ## Note - right canonical form assumed in beginning, and left canonical form in the end
    assert len(Us) == M.L, "Length of Us must match the number of sites in M"
    #print(len(Us), M.L)
    N = M.L
    normS = 1.
    Uind = 0
    for site in range(N):
        U = Us[Uind]
        #print(site)
        norm_site, M = evol_one_site(M, U, site, "l", truncation = truncation, max_bd = max_bd, max_err = max_err)
        normS *= norm_site
        Uind += 1
    return normS, M



def _strip_size1_outer_bonds(M):
    """Remove size-1 outer (dangling, non-shared) bonds from an MPS in place.

    The existing compressed-forward path re-adds explicit boundary-bond indices
    like `Boundary_L{T-1}_S{0}` (see the end of Pauli_MPS_after_QMLM_output_only).
    Meanwhile the bath-traced target MPS has no such bonds, causing
    `tensor_network_distance` to reject the pair. Stripping size-1 outer bonds
    from both sides reconciles the outer-index sets without changing the
    represented operator (a size-1 unnamed leg is just a trivial direct sum).
    """
    N_sites = M.L
    for i in range(N_sites):
        t = M[i]
        shared = set()
        if i > 0:
            shared |= set(t.inds) & set(M[i - 1].inds)
        if i < N_sites - 1:
            shared |= set(t.inds) & set(M[i + 1].inds)
        outer = [idx for idx in t.inds if idx not in shared]
        drop = [idx for idx in outer if t.ind_size(idx) == 1]
        if not drop:
            continue
        remaining_inds = [idx for idx in t.inds if idx not in drop]
        # Transpose so the to-drop axes are leading, then squeeze.
        t.transpose_(*drop, *remaining_inds)
        new_data = t.data
        for _ in drop:
            new_data = new_data.squeeze(0)
        t.modify(data=new_data, inds=remaining_inds)
    return M



def _qr_compress_cyclic_to_obc(M, layer_idx, max_bd, max_err):
    """Per-layer bond compression using QR-based (``method="direct"``) 1-D MPS
    compression. QR has a stable autograd backward, unlike SVD which yields
    NaN through ``svd_backward`` on degenerate singular values.

    Current quimb's ``tensor_network_1d_compress`` drops the size-1 virtual
    bonds at sites 0 / L-1. That breaks subsequent ``one_site_layer`` calls
    (which assume 3-D tensors) and the cyclic boundary bond shared by sites
    0 and L-1 (the left_canonicalize inside ``evol_one_site`` then fails
    with "tensors don't share a bond"). We restore the 3-D shape with a
    shared boundary index name and flip ``cyclic`` to False so subsequent
    canonicalize calls treat the MPS as OBC.
    """
    for j_inner in range(M.L):
        inds = list(M[j_inner].inds)
        inds[1], inds[2] = inds[2], inds[1]
        M[j_inner].transpose_(*inds)
    M = qtn.tensor_network_1d_compress(
        M, method="direct", max_bond=max_bd, cutoff=max_err,
        cutoff_mode="rsum2", permute_arrays=False,
    )
    boundary_name = f"Boundary_L{layer_idx}"
    for j_inner in range(M.L):
        t = M[j_inner]
        if j_inner == 0 and len(t.inds) == 2:
            new_data = t.data.reshape(1, *t.data.shape)
            M[j_inner].modify(data=new_data, inds=(boundary_name, t.inds[0], t.inds[1]))
        elif j_inner == M.L - 1 and len(t.inds) == 2:
            new_data = t.data.reshape(t.data.shape[0], 1, t.data.shape[1])
            M[j_inner].modify(data=new_data, inds=(t.inds[0], boundary_name, t.inds[1]))
    for j_inner in range(M.L):
        inds = list(M[j_inner].inds)
        inds[1], inds[2] = inds[2], inds[1]
        M[j_inner].transpose_(*inds)
    M.cyclic = False
    return M



def Pauli_MPS_after_QMLM_output_only(M, T, all_U, p_depolar=0, p_dephaseX=0, p_dephaseY=0, p_dephaseZ=0, max_bd = 1024, max_err = 1E-10, truncation = False, noise_type = "all", all_tms=None):
    """ Run a circuit with T layers of random unitaries and depolarization noise,
    starting from a random Pauli MPS.
    Parameters:
        M (qtn.MatrixProductState): Initial MPS with Pauli operators.
        T (int): Number of layers to apply.
        p (float): Probability of applying depolarization noise.    
    Returns:
        M (qtn.MatrixProductState): Final MPS after T layers of evolution.
    """
    L = M.L
    assert len(all_U) == T, "Length of all_U must match the number of layers T. It has {}".format(len(all_U))
    '''assert len(p_dephaseX) == T, "Length of dephasing error X must match the number of layers T. It has {}".format(len(p_dephaseX))
    assert len(p_dephaseY) == T, "Length of dephasing error Y must match the number of layers T. It has {}".format(len(p_dephaseY))
    assert len(p_dephaseZ) == T, "Length of dephasing error Z must match the number of layers T. It has {}".format(len(p_dephaseZ))
    assert len(p_depolar) == T, "Length of depolarization error must match the number of layers T. It has {}".format(len(p_depolar))
    M.astype_('float64')'''
    # Promote every site tensor to a single common dtype, at least float64
    # (the transfer matrices applied below are float64). Raw Pauli inputs from
    # Identity_init are float32 and gates promote only the sites they touch,
    # so at odd N a float32 tail site survives to the first
    # right_canonicalize in the "all" branch and crashes with
    # "both inputs should have same dtype". torch.promote_types never loses
    # information (float32→float64, real→complex), and callers that already
    # pre-cast (all training loops) hit the no-op path.
    _common = torch.float64
    for _j in range(L):
        _common = torch.promote_types(_common, M[_j].data.dtype)
    for _j in range(L):
        if M[_j].data.dtype != _common:
            M[_j].modify(data=M[_j].data.to(_common))
    normS = 1.
    if noise_type == "all":
        for i in range(T):
            Noise_list_depolar = [depolarization_noise_transfer_matrix(p_depolar[i, j]) for j in range(L)]
            Noise_list_dephasingX = [dephasing_noise_transfer_matrix(p_dephaseX[i, j], _DIR_X) for j in range(L)]
            Noise_list_dephasingY = [dephasing_noise_transfer_matrix(p_dephaseY[i, j], _DIR_Y) for j in range(L)]
            Noise_list_dephasingZ = [dephasing_noise_transfer_matrix(p_dephaseZ[i, j], _DIR_Z) for j in range(L)]
            U_list = all_U[i,:L]
            tm_list = all_tms[i, :L] if all_tms is not None else None


            norm_i, M = two_site_layer(M, U_list[:L//2], "o", truncation = truncation, max_bd = max_bd, max_err = max_err, tms=(tm_list[:L//2] if tm_list is not None else None))
            M.right_canonicalize()

            norm_i, M = two_site_layer(M, U_list[L//2:], "e", truncation = truncation, max_bd = max_bd, max_err = max_err, tms=(tm_list[L//2:] if tm_list is not None else None))

            norm_i, M = one_site_layer(M, Noise_list_depolar, max_bd = max_bd, max_err = max_err)

            norm_i, M = one_site_layer(M, Noise_list_dephasingX, max_bd = max_bd, max_err = max_err)

            norm_i, M = one_site_layer(M, Noise_list_dephasingY, max_bd = max_bd, max_err = max_err)

            norm_i, M = one_site_layer(M, Noise_list_dephasingZ, max_bd = max_bd, max_err = max_err)
            M.right_canonicalize()

            if truncation:
                M = _qr_compress_cyclic_to_obc(M, i, max_bd, max_err)
    elif noise_type == "depolarizing":
        for i in range(T):
            Noise_list_depolar = [depolarization_noise_transfer_matrix(p_depolar[i, j]) for j in range(L)]
            U_list = all_U[i,:L]
            tm_list = all_tms[i, :L] if all_tms is not None else None

            norm_i, M = two_site_layer(M, U_list[:L//2], "o", truncation = truncation, max_bd = max_bd, max_err = max_err, tms=(tm_list[:L//2] if tm_list is not None else None))

            norm_i, M = two_site_layer(M, U_list[L//2:], "e", truncation = truncation, max_bd = max_bd, max_err = max_err, tms=(tm_list[L//2:] if tm_list is not None else None))
            norm_i, M = one_site_layer(M, Noise_list_depolar, max_bd = max_bd, max_err = max_err)
            if truncation:
                M = _qr_compress_cyclic_to_obc(M, i, max_bd, max_err)
    elif noise_type == "dephasing":
        for i in range(T):
            Noise_list_dephasingX = [dephasing_noise_transfer_matrix(p_dephaseX[i, j], _DIR_X) for j in range(L)]
            Noise_list_dephasingY = [dephasing_noise_transfer_matrix(p_dephaseY[i, j], _DIR_Y) for j in range(L)]
            Noise_list_dephasingZ = [dephasing_noise_transfer_matrix(p_dephaseZ[i, j], _DIR_Z) for j in range(L)]
            U_list = all_U[i,:L]
            tm_list = all_tms[i, :L] if all_tms is not None else None

            norm_i, M = two_site_layer(M, U_list[:L//2], "o", truncation = truncation, max_bd = max_bd, max_err = max_err, tms=(tm_list[:L//2] if tm_list is not None else None))

            norm_i, M = two_site_layer(M, U_list[L//2:], "e", truncation = truncation, max_bd = max_bd, max_err = max_err, tms=(tm_list[L//2:] if tm_list is not None else None))

            norm_i, M = one_site_layer(M, Noise_list_dephasingX, max_bd = max_bd, max_err = max_err)
            norm_i, M = one_site_layer(M, Noise_list_dephasingY, max_bd = max_bd, max_err = max_err)
            norm_i, M = one_site_layer(M, Noise_list_dephasingZ, max_bd = max_bd, max_err = max_err)

            if truncation:
                M = _qr_compress_cyclic_to_obc(M, i, max_bd, max_err)
    else:
        raise ValueError(f"noise_type are all, depolarizing, dephasing. Not: {noise_type}")
    return M


def ensure_complex_torch(tn, dtype=torch.complex64):
    tn = tn.copy()
    tn.apply_to_arrays(lambda x: torch.as_tensor(x).to(dtype))
    return tn


def sanitize_mps(M):
    for tensor in M:
        data = tensor.data
        if not np.all(np.isfinite(data)):
            data[~np.isfinite(data)] = 0.0
        norm = np.linalg.norm(data)
        if norm > 0:
            tensor.modify(data=data / norm)
    return M

