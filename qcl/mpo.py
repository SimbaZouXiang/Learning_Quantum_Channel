"""MPO-building forward path (exact but exponential in circuit depth).

Builds a quimb tensor-network MPO for the whole circuit and merges it with an
input MPS via ``mpo | mps``. Prefer the compressed layer-by-layer path in
``qcl.evolve`` for deep circuits.
"""
import torch
import quimb.tensor as qtn

from .pauli import construct_SU4_from_input, unitary_to_transfer_matrix_two_site_reshape
from .noise import (depolarization_noise_transfer_matrix,
                    dephasing_noise_transfer_matrix_X,
                    dephasing_noise_transfer_matrix_Y,
                    dephasing_noise_transfer_matrix_Z)


def Build_2_site_MPO_from_transfer_matrices(U, d1=4, d2=4, i = 0, tags = 'T'):
    """
    Build an MPO from a list of transfer matrices.

    Parameters:
        T (list of np.ndarray): List of transfer matrices, each of shape (4, 4)."""
    mpo = qtn.PTensor(
    fn=unitary_to_transfer_matrix_two_site_reshape,
    params=U,
    inds=[f'b{i}', f'b{i+1}', f'k{i}', f'k{i+1}'],
    tags=tags,
)
    
    return mpo


def First_identity_layer(N):
    """Generate the first identity layer of the Operators"""
    """
        N: int
        The number of qubits
    """
    mpo = qtn.Tensor(torch.diag(torch.ones(4)), inds=['input0','k0'], tags='Id')
    TN = mpo
    for site in range(1, N):
        mpo = qtn.Tensor(torch.diag(torch.ones(4)), inds=[f'input{site}', f'k{site}'], tags='Id')
        TN = TN | mpo
    return TN


def Apply_one_site_layer(N, TN, form = "depolarizing", p=0, layer = 0, cdtype = 'float64'):
    # Dispatch to the correct noise function
    _NOISE_FN = {
        "depolarizing": depolarization_noise_transfer_matrix,
        "dephasingX": dephasing_noise_transfer_matrix_X,
        "dephasingY": dephasing_noise_transfer_matrix_Y,
        "dephasingZ": dephasing_noise_transfer_matrix_Z,
    }
    fn = _NOISE_FN.get(form)
    if fn is None:
        raise ValueError(f"Noise form are depolarizing, dephasingX, dephasingY, dephasingZ. Not: {form}")

    mpo = qtn.PTensor(fn=fn, params=p[0], inds=['k0', 'b0'], tags=f"{form}_{layer}")
    for site in range(1, N):
        this_mpo = qtn.PTensor(fn=fn, params=p[site], inds=[f'k{site}', f'b{site}'], tags=f"{form}_{layer}")
        mpo = mpo | this_mpo
    mpo = mpo.astype(cdtype)
    TN = TN | mpo
    for site in range(N):
        fixed_uuid = f"1site_{form}_L{layer}_S{site}"
        TN.reindex_({f'k{site}': fixed_uuid})
        TN.reindex_({f'b{site}': f'k{site}'})

    return TN



def Apply_two_site_layer(N, TN, U_list, eo = "o", d1=4, d2=4, layer = 0):
    ## Apply two site operator at each even/odd sites
    ## Note - right canonical form assumed in beginning, and left canonical form in the end
    ## Us = 2-site unitary of length N/2 ("o") or N/2-1 ("e") 
    Uind = 0
    cdtype  = 'complex64' # or 'complex128' if desired
    if(eo == "o"):
        site0 = 0
    else:
        site0 = 1
    TN = TN.astype(cdtype)
    U = U_list[Uind]
    mpo = Build_2_site_MPO_from_transfer_matrices(U, i = site0, tags = f'T_{layer}_{eo}')
    mpo = mpo.astype(cdtype)
    Uind += 1

    for site in range(site0+2, N-1, 2):
        U = U_list[Uind]
        this_mpo = Build_2_site_MPO_from_transfer_matrices(U, i = site, tags = f'T_{layer}_{eo}')
        this_mpo = this_mpo.astype(cdtype)
        mpo = mpo | this_mpo
        Uind += 1
    TN = TN | mpo
    for site in range(site0, N-1, 2):
        fixed_uuid_1 = f"2site_{eo}_L{layer}_S{site}"
        fixed_uuid_2 = f"2site_{eo}_L{layer}_S{site+1}"
        TN.reindex_({f'k{site}': fixed_uuid_1, f'k{site+1}': fixed_uuid_2})
        TN.reindex_({f'b{site}': f'k{site}', f'b{site+1}': f'k{site+1}'})

    return TN


def Build_QMLM_MPO(N, T, all_U, p_depolar=0, p_dephaseX=0, p_dephaseY=0, p_dephaseZ=0, noise_type = "all"):
    """ Run a circuit with T layers of random unitaries and depolarization noise, returning the exact circuit.
    starting from a random Pauli MPS.
    Parameters:
        M (qtn.MatrixProductState): Initial MPS with Pauli operators.
        T (int): Number of layers to apply.
        p (float): Probability of applying depolarization noise.  
        noise_type (str): choose from "all", "depolarizing", "dephasing", "none"  
    Returns:
        M (qtn.MatrixProductState): Final MPS after T layers of evolution.
    """
    assert len(all_U) == T, "Length of all_U must match the number of layers T. It has {}".format(len(all_U))
    TN = First_identity_layer(N)
    
    if noise_type == "all":
        assert len(p_dephaseX) == T, "Length of dephasing error X must match the number of layers T. It has {}".format(len(p_dephaseX))
        assert len(p_dephaseY) == T, "Length of dephasing error Y must match the number of layers T. It has {}".format(len(p_dephaseY))
        assert len(p_dephaseZ) == T, "Length of dephasing error Z must match the number of layers T. It has {}".format(len(p_dephaseZ))
        assert len(p_depolar) == T, "Length of depolarization error must match the number of layers T. It has {}".format(len(p_depolar))
        for layer in range(T):
            U_list = all_U[layer,:N]
            TN = Apply_two_site_layer(N, TN, U_list[:N//2], eo = "o", layer = layer)
            TN = Apply_two_site_layer(N, TN, U_list[N//2:], eo = "e", layer = layer)
            TN = Apply_one_site_layer(N, TN, form = "depolarizing", p=p_depolar[layer], layer = layer)
            TN = Apply_one_site_layer(N, TN, form = "dephasingX", p=p_dephaseX[layer], layer = layer)
            TN = Apply_one_site_layer(N, TN, form = "dephasingY", p=p_dephaseY[layer], layer = layer)
            TN = Apply_one_site_layer(N, TN, form = "dephasingZ", p=p_dephaseZ[layer], layer = layer)
        return TN
    elif noise_type == "none":
        for layer in range(T):
            U_list = all_U[layer,:N]
            TN = Apply_two_site_layer(N, TN, U_list[:N//2], eo = "o", layer = layer)
            TN = Apply_two_site_layer(N, TN, U_list[N//2:], eo = "e", layer = layer)
        return TN
    elif noise_type == "depolarizing":
        assert len(p_depolar) == T, "Length of depolarization error must match the number of layers T. It has {}".format(len(p_depolar))
        for layer in range(T):
            U_list = all_U[layer,:N]
            TN = Apply_two_site_layer(N, TN, U_list[:N//2], eo = "o", layer = layer)
            TN = Apply_two_site_layer(N, TN, U_list[N//2:], eo = "e", layer = layer)
            TN = Apply_one_site_layer(N, TN, form = "depolarizing", p=p_depolar[layer], layer = layer)
        return TN
    elif noise_type == "dephasing":
        for layer in range(T):
            U_list = all_U[layer,:N]
            TN = Apply_two_site_layer(N, TN, U_list[:N//2], eo = "o", layer = layer)
            TN = Apply_two_site_layer(N, TN, U_list[N//2:], eo = "e", layer = layer)
            TN = Apply_one_site_layer(N, TN, form = "dephasingX", p=p_dephaseX[layer], layer = layer)
            TN = Apply_one_site_layer(N, TN, form = "dephasingY", p=p_dephaseY[layer], layer = layer)
            TN = Apply_one_site_layer(N, TN, form = "dephasingZ", p=p_dephaseZ[layer], layer = layer)
        return TN
    else:
        raise ValueError(f"noise_type are all, depolarizing, dephasing or none. Not: {noise_type}")


def Pauli_MPS_after_QMLM(M, TN):
    M_out = M | TN
    return M_out



def compress_tn_to_mps(tn, N, max_bond=64, cutoff=1e-10):
    """Compress a merged tensor network (e.g. ``mpo | mps``) into a proper MPS.

    This is a convenience utility for the common pattern::

        M_merged = mpo_fit | input_mps   # large uncontracted TN
        M_compressed = compress_tn_to_mps(M_merged, N, max_bond=64)
        loss = tensor_network_distance(M_compressed, target_mps)  # fast

    It uses quimb's 1-D compression (qr-based direct method) to turn
    the merged TN into an MPS whose bond dimension is at most *max_bond*.

    **Prefer** ``QMLM.forward_compressed`` or ``apply_mpo_to_mps_compressed``
    when possible — they apply the circuit layer-by-layer which is both more
    numerically stable and faster for deep circuits.  This helper is useful as
    a quick drop-in when you already have a merged TN.

    Parameters:
        tn: quimb TensorNetwork (result of ``mpo | mps``).
        N (int): Number of physical sites.
        max_bond (int): Maximum bond dimension.
        cutoff (float): qr truncation cutoff.

    Returns:
        qtn.MatrixProductState: Compressed MPS with physical indices 'k0'…'k{N-1}'.
    """
    site_ind_id = 'k{}'
    outer_inds = [site_ind_id.format(i) for i in range(N)]
    try:
        # quimb ≥ 1.8: contract_compressed builds the MPS directly
        mps = tn.contract_compressed(
            output_inds=outer_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode='rsum2',
        )
    except AttributeError:
        # Fallback: contract everything then do an qr/qr sweep
        raw = tn.contract(output_inds=outer_inds)
        # raw is a single big tensor; decompose into MPS form
        mps = qtn.MatrixProductState.from_dense(
            raw.data, dims=[4] * N,
            max_bond=max_bond, cutoff=cutoff,
            method='qr'
        )
    return mps

