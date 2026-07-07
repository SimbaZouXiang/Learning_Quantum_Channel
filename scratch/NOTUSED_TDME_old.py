import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
import time
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.fitting import tensor_network_distance
from ncon import ncon
import torch
import warnings
import torch.optim as optim
from quimb.tensor import tensor_split
import cotengra as ctg
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ──────────────────────────────────────────────────────────────────────
# Module-level cached constants (computed once at import time)
# ──────────────────────────────────────────────────────────────────────

def _build_pauli_basis():
    """Build and cache the (2,2,4) Pauli basis tensor once."""
    dtype = torch.complex128
    I = torch.eye(2, dtype=dtype)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    Ps = torch.stack([I, X, Y, Z], dim=-1)
    Ps.requires_grad = False
    return Ps

_PAULI_BASIS = _build_pauli_basis()                        # shape (2, 2, 4)
_PAULI_BASIS_2SITE = torch.einsum(
    'a b c, d e f -> a d b e c f', _PAULI_BASIS, _PAULI_BASIS
).reshape(4, 4, 16)                                        # shape (4, 4, 16)

# Precomputed single-Pauli transfer matrices: σ P_i σ† → T_{ij}
# For Pauli σ_k (k=1,2,3), the transfer matrix is diag with ±1 entries.
_DEPHASING_TM_X = torch.diag(torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64))
_DEPHASING_TM_Y = torch.diag(torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64))
_DEPHASING_TM_Z = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))

# Precomputed direction tensors for dephasing noise
_DIR_X = torch.tensor([0, 1, 0, 0], dtype=torch.float64)
_DIR_Y = torch.tensor([0, 0, 1, 0], dtype=torch.float64)
_DIR_Z = torch.tensor([0, 0, 0, 1], dtype=torch.float64)

# Cache identity for noise matrices
_EYE4 = torch.eye(4, dtype=torch.float64)


def Identity_init(L, bond_dim=1, phys_dim=4):
    """
    Generate an MPS representing the product state |0>^{⊗L}
    with specified bond and physical dimensions.

    Parameters:
        L (int): Number of sites.
        bond_dim (int): Desired bond dimension (>=1).
        phys_dim (int): Physical dimension at each site.

    Returns:
        qtn.MatrixProductState: The identity/product MPS.
    """
    tensors = []
    for i in range(L):
        # Each site tensor: shape (left_bond, phys_dim, right_bond)
        left = 1 if i == 0 else bond_dim
        right = 1 if i == L - 1 else bond_dim
        data = torch.zeros((left, phys_dim, right), requires_grad=False)
        # Set the |0> component to 1 along the diagonal bonds
        for b in range(min(left, right)):
            data[b, 0, b] = 1.0
        tensors.append(data)
    M = qtn.MatrixProductState(tensors) 
    for i, tensor in enumerate(M):
        indices = tensor.inds
        M[i].reindex({indices[1]: indices[2], indices[2]: indices[1]}, inplace=True)
        M[i].reindex_({f'k{i}': f'input{i}'})
    M[i].reindex({indices[2]: 'void2'}, inplace=True)

    return M

def operator_assignment_single_site(M, site, operator):
    """
    Assign an operator to a single site in an MPS.

    Parameters:
        M (qtn.MatrixProductState): The MPS to modify.
        site (int): The index of the site to assign the operator to.
        operator (np.ndarray): The operator to assign, shape (phys_dim, phys_dim).

    Returns:
        qtn.MatrixProductState: The modified MPS with the operator assigned.
    """
    # Create a new tensor for the specified site
    N = M.L
    assert 0 <= site <= N-1
    if operator == "X":
        M[site].modify(data = torch.tensor([0.0, 1.0, 0.0, 0.0]).reshape(1, 4, 1))
    elif operator == "Y":
        M[site].modify(data = torch.tensor([0.0, 0.0, 1.0, 0.0]).reshape(1, 4, 1))
    elif operator == "Z":
        M[site].modify(data = torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 4, 1))
    else:
        raise ValueError(f"Operator not defined! Site: {site}, Operator: {operator}")
    
    return M

def operator_assignment(M, sites, operators):
    """ Assign operators to multiple sites in an MPS.
    Parameters:
        M (qtn.MatrixProductState): The MPS to modify.
        sites (list of int): Indices of the sites to assign operators to.
        operators (list of str): Operators to assign at each site, e.g., ["X", "Y", "Z"].
    Returns:
        qtn.MatrixProductState: The modified MPS with operators assigned.
    """
    assert len(sites) == len(operators)
    for i in range(len(sites)):
        site = sites[i]
        operator = operators[i]
        M = operator_assignment_single_site(M, site, operator)
    return M


def Haar_random_unitary(d):
    """Return a dxd Haar random unitary matrix (complex)"""
    H = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    Q, R = np.linalg.qr(H)
    fac = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(d)])
    Q = Q @ fac
    Q = torch.from_numpy(Q)
    return Q

@torch.no_grad()
def Pauli_operator_basis(device=None):
    """Return cached (2,2,4) Pauli basis tensor."""
    return _PAULI_BASIS

def unitary_to_transfer_matrix_single_site(U):
    """ Convert a single-site unitary to a transfer matrix.
    Parameters:
        U (np.ndarray): 2x2 unitary matrix. 
    Returns:    
        np.ndarray: Real part of the transfer matrix T of shape (4, 4).
    """
    Ps = _PAULI_BASIS
    Uc = U.conj().T
    T = torch.einsum('ab,bci,cd,daj->ij', U, Ps, Uc, Ps) / 2
    return torch.real(T)

def unitary_to_transfer_matrix_two_site(U):
    """
    Generalization to 2-site unitary.

    Parameters:
        U (np.ndarray): 4x4 unitary matrix.

    Returns:
        np.ndarray: Real part of the transfer matrix T of shape (16, 16).
    """
    U = construct_SU4_from_input(U.reshape(16,))  # ensure U is unitary
    Ps2 = _PAULI_BASIS_2SITE
    Uc = U.conj().T
    T = torch.einsum('a b, b c i, c d, d a j -> i j', U, Ps2, Uc, Ps2) / 4
    return torch.real(T)


def unitary_to_transfer_matrix_two_site_reshape(U, d=4):
    """
    Reshape the transfer matrix for a 2-site unitary.

    Parameters:
        U (np.ndarray): 4x4 unitary matrix.

    Returns:
        np.ndarray: Real part of the reshaped transfer matrix T of shape (16, 16).
    """
    T = unitary_to_transfer_matrix_two_site(U)
    return T.reshape(d, d, d, d)

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
        TN.reindex_({f'k{site}': qtn.rand_uuid()})
        TN.reindex_({f'b{site}': f'k{site}'})

    return TN

def dephasing_noise_transfer_matrix(p, dir):
    """
    Generate the transfer matrix for a dephasing noise channel.
    Parameters:
        p (float): Probability of applying the noise.
        dir (list of float): Direction vector for the noise, e.g., [0, 1, 0, 0] for X direction.
    
    Returns:
        np.ndarray: The transfer matrix of shape (4, 4)."""
    Ps = _PAULI_BASIS
    sigma = dir[1] * Ps[:,:,1] + dir[2] * Ps[:,:,2] + dir[3] * Ps[:,:,3]
    Tp = (p/2) * unitary_to_transfer_matrix_single_site(sigma)
    Ti = (1-p/2) * _EYE4
    return (Ti+Tp)

def dephasing_noise_transfer_matrix_X(p):
    """Generate dephasing noise transfer matrix for X direction using precomputed TM."""
    Ti = (1-p/2) * _EYE4
    Tp = (p/2) * _DEPHASING_TM_X
    return (Ti+Tp)

def dephasing_noise_transfer_matrix_Y(p):
    """Generate dephasing noise transfer matrix for Y direction using precomputed TM."""
    Ti = (1-p/2) * _EYE4
    Tp = (p/2) * _DEPHASING_TM_Y
    return (Ti+Tp)

def dephasing_noise_transfer_matrix_Z(p):
    """Generate dephasing noise transfer matrix for Z direction using precomputed TM."""
    Ti = (1-p/2) * _EYE4
    Tp = (p/2) * _DEPHASING_TM_Z
    return (Ti+Tp)

def depolarization_noise_transfer_matrix(p):
    
    coeff = 1.0 - 4.0 * p / 3.0
    
    # Create matrix from scratch with proper gradient tracking
    T = torch.zeros((4, 4), dtype=torch.float64)
    T[0, 0] = 1.0
    T[1, 1] = coeff
    T[2, 2] = coeff  
    T[3, 3] = coeff
    
    return T

def random_pauli_string(n):
    """
    Generate a random Pauli string of length n.
    
    Parameters:
        n (int): Length of the Pauli string.
    
    Returns:
        str: Random Pauli string consisting of 'I', 'X', 'Y', 'Z'.
    """
    pauli_ops = ['I', 'X', 'Y', 'Z']
    return ''.join(np.random.choice(pauli_ops) for _ in range(n))

def random_pauli_MPS(n, weight = None):
    """
    Generate a random MPS with Pauli operators.
    
    Parameters:
        n (int): Number of sites in the MPS.
    
    Returns:
        qtn.MatrixProductState: Random MPS with Pauli operators.
    """
    if weight is None:
        weight = np.random.randint(1, n+1)
        M = Identity_init(n, bond_dim=1, phys_dim=4)
        idx_to_change = np.random.choice(range(n), size=weight, replace=False)
        operator_list = []
        for idx in idx_to_change:
            operator_list.append(np.random.choice(["X", "Y", "Z"]))
        M = operator_assignment(M, idx_to_change, operator_list)
        return M, weight
    else:
        M = Identity_init(n, bond_dim=1, phys_dim=4)
        idx_to_change = np.random.choice(range(n), size=weight, replace=False)
        operator_list = []
        for idx in idx_to_change:
            operator_list.append(np.random.choice(["X", "Y", "Z"]))
        M = operator_assignment(M, idx_to_change, operator_list)
        return M, weight
        
    
def construct_SU4_from_input(params, device='cpu'):
    """ Construct a SU(4) unitary from 16 parameters.

    Parameters:
        params (np.ndarray): Array of shape (T, 16) containing the parameters for the SU(4) generators.
    Returns:
        np.ndarray: Array of shape (T, 4, 4) containing the SU(4) unitaries.
    """
    assert len(params)==16, "need 16 parameters for SU(4) construction, input has {}".format(len(params))
    Q, R = torch.linalg.qr(torch.reshape(params, (4, 4)))
    diag_R = torch.diagonal(R)
    phases = torch.where(
        diag_R.abs() > 0,
        diag_R / diag_R.abs(),
        torch.ones_like(diag_R),
    )
    fac = torch.diag(phases.conj())  
    Q = Q @ fac  # still unitary
    return Q

def construct_all_SU4(L, T, param_list):
    """ Construct a list of SU(4) unitaries for each layer in the circuit.

    Parameters:
        L (int): Number of sites in the MPS.
        T (int): Number of layers in the circuit.
        param_list: parameters list of shape (T, L-1, 15) containing the parameters for each layer.

    Returns:
        list: List of SU(4) unitaries for each layer, each of shape (L-1, 4, 4).
    """
    assert param_list.shape == (T, L, 16), "Parameter list length must match the number of layers T."
    all_U = torch.zeros((T, L, 4, 4), dtype=torch.complex128, requires_grad=False) #!
    for i in range(T):
        for j in range(L-1):
            U = construct_SU4_from_input(param_list[i][j])
            all_U[i, j] = U  # Repeat U for each site
    return all_U


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
        TN.reindex_({f'k{site}': qtn.rand_uuid(), f'k{site+1}': qtn.rand_uuid()})
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

class QMLM(nn.Module):
    def __init__(self, N, T, param = None, p_depolar = None, p_dephaseX = None, p_dephaseY = None, p_dephaseZ = None, max_bd = 1024, max_err = 1E-10):
        super(QMLM, self).__init__()
        if p_depolar is None:
            p_depolar = torch.zeros((T, N), dtype=torch.float64, requires_grad=True)
        if p_dephaseX is None:
            p_dephaseX = torch.zeros((T, N), dtype=torch.float64)
        if p_dephaseY is None:
            p_dephaseY = torch.zeros((T, N), dtype=torch.float64)
        if p_dephaseZ is None:
            p_dephaseZ = torch.zeros((T, N), dtype=torch.float64)
        self.layers = T
        self.N = N
        if param is None:
            params = torch.zeros(T, N, 16, dtype=torch.complex128)
            for t in range(T):
                for n in range(N):
                    identity_param = torch.eye(4, dtype=torch.complex128)
                    params[t, n] = identity_param.reshape(16,)
            self.params = nn.Parameter(params, requires_grad=True)
        else:
            assert param.shape == (T, N, 16), "Parameter list length must match the number of layers T and N."
            self.params = param
        self.p_depolar = nn.Parameter(p_depolar, requires_grad=True)
        self.p_dephaseX = nn.Parameter(p_dephaseX, requires_grad=True)
        self.p_dephaseY = nn.Parameter(p_dephaseY, requires_grad=True)
        self.p_dephaseZ = nn.Parameter(p_dephaseZ, requires_grad=True)
        self.max_bd = max_bd
        self.max_err = max_err


    def forward(self, M_input):
        """ Forward pass of the QMLM model.
        
        Parameters:
            M_input (qtn.MatrixProductState): Input MPS.
        
        Returns:
            qtn.MatrixProductState: Output MPS after applying the QMLM model.
        """
        #with torch.autograd.detect_anomaly():
        all_U = construct_all_SU4(self.N, self.layers, self.params)
        TN = Build_QMLM_MPO(self.N, self.layers, all_U, (self.p_depolar), self.p_dephaseX, self.p_dephaseY, self.p_dephaseZ, noise_type = "all")
        M_output = Pauli_MPS_after_QMLM(M_input, TN)

        return M_output  # Return the output MPS after applying the QMLM model
    
    def loss(self, M_input, M_realoutput, i = 1, flag = False):
        """
        Calculate a stable, differentiable loss for the QMLM model using
        the squared Frobenius distance between MPS states.

        Returns:
            torch.Tensor: Scalar loss (requires_grad=True).
        """
        M_output = self.forward(M_input)
        M_output.astype_('complex128')
        M_realoutput.astype_('complex128')
        if i%50 == 0 and flag:
            print(M_output.H @ M_output, M_realoutput.H @ M_realoutput)
        return tensor_network_distance(M_output, M_realoutput)
    
    def forward_without_noise(self, M_input):
        """ Forward pass of the QMLM model.
        
        Parameters:
            M_input (qtn.MatrixProductState): Input MPS.
        
        Returns:
            qtn.MatrixProductState: Output MPS after applying the QMLM model.
        """
        #with torch.autograd.detect_anomaly():
        #all_U = construct_all_SU4(self.N, self.layers, self.params)
        TN = Build_QMLM_MPO(self.N, self.layers, self.params, noise_type = "none")
        M_output = Pauli_MPS_after_QMLM(M_input, TN)

        return M_output  # Return the output MPS after applying the QMLM model
    
    def loss_without_noise(self, M_input, M_realoutput, i = 1):
        """
        Calculate a stable, differentiable loss for the QMLM model using
        the squared Frobenius distance between MPS states.

        Returns:
            torch.Tensor: Scalar loss (requires_grad=True).
        """
        # Forward
        M_output = self.forward_without_noise(M_input)
        M_output.astype_('complex128')
        M_realoutput.astype_('complex128')
        return tensor_network_distance(M_output, M_realoutput)

    def get_MPO(self, noise_type = "all"):
        #all_U = construct_all_SU4(self.N, self.layers, self.params)
        TN = Build_QMLM_MPO(self.N, self.layers, self.params, self.p_depolar, self.p_dephaseX, self.p_dephaseY, self.p_dephaseZ, noise_type = noise_type)
        return TN

    def forward_compressed(self, M_input, max_bond=64, cutoff=1e-10, noise_type="all"):
        """Apply the QMLM circuit layer-by-layer to an MPS with bond-dimension truncation.

        Instead of building the full MPO tensor network and merging it with the MPS
        (which creates a large 2D-like TN whose contraction scales exponentially
        with circuit depth), this method applies each circuit layer directly to the
        MPS and optionally compresses after each layer, keeping the bond dimension
        bounded by *max_bond*.

        The subsequent call to ``tensor_network_distance`` then operates on two
        proper MPS objects, reducing the cost from exponential to O(N * D^3).

        Parameters:
            M_input (qtn.MatrixProductState): Input MPS (will be copied internally).
            max_bond (int): Maximum bond dimension after compression.
            cutoff (float): Singular value cutoff for truncation.
            noise_type (str): "all", "depolarizing", "dephasing", or "none".

        Returns:
            qtn.MatrixProductState: Output MPS with bounded bond dimension.
        """
        all_U = construct_all_SU4(self.N, self.layers, self.params)
        M_output = Pauli_MPS_after_QMLM_output_only(
            M_input.copy(), self.layers, all_U,
            self.p_depolar, self.p_dephaseX, self.p_dephaseY, self.p_dephaseZ,
            max_bd=max_bond, max_err=cutoff, truncation=True, noise_type=noise_type,
        )
        return M_output

    def forward_compressed_depolarizing_only(self, M_input, max_bond=64, cutoff=1e-10):
        """Same as forward_compressed but with depolarizing noise only."""
        all_U = construct_all_SU4(self.N, self.layers, self.params)
        M_output = Pauli_MPS_after_QMLM_output_only(
            M_input.copy(), self.layers, all_U,
            self.p_depolar,
            max_bd=max_bond, max_err=cutoff, truncation=True, noise_type="depolarizing",
        )
        return M_output


def apply_mpo_to_mps_compressed(model, input_mps_list, max_bond=64, cutoff=1e-10, noise_type="all"):
    """Apply the QMLM model to a list of input MPS, returning compressed output MPS.

    This is more efficient than the pattern ``model.get_MPO() | mps`` because:
    1. The circuit is applied layer-by-layer with SVD truncation, keeping
       the bond dimension at most *max_bond*.
    2. The returned objects are proper MPS — ``tensor_network_distance``
       between two MPS is O(N D^3), vs exponential for an uncontracted TN.

    The SU(4) unitaries are constructed once and re-used for every sample.

    Parameters:
        model (QMLM): Trained QMLM model.
        input_mps_list (list): List of input MPS.
        max_bond (int): Maximum bond dimension after compression.
        cutoff (float): SVD truncation cutoff.
        noise_type (str): "all", "depolarizing", "dephasing", or "none".

    Returns:
        list: List of compressed output MPS (with physical indices 'k0','k1',…).
    """
    all_U = construct_all_SU4(model.N, model.layers, model.params)
    output_list = []
    for mps in input_mps_list:
        M_out = Pauli_MPS_after_QMLM_output_only(
            mps.copy(), model.layers, all_U,
            model.p_depolar, model.p_dephaseX, model.p_dephaseY, model.p_dephaseZ,
            max_bd=max_bond, max_err=cutoff, truncation=True, noise_type=noise_type,
        )
        # Reindex from 'input{i}' to 'k{i}' to match targets
        for i, tensor in enumerate(M_out):
            tensor.reindex_({f'input{i}': f'k{i}'})
        output_list.append(M_out)
    return output_list


# ---------------Below are for output MPS only!!!!!!!!!!!!!!!!-----------------------



def unitary_to_transfer_matrix_single_site_truncated(U):
    """ Convert a single-site unitary to a transfer matrix.
    Parameters:
        U (np.ndarray): 2x2 unitary matrix. 
    Returns:    
        np.ndarray: Real part of the transfer matrix T of shape (4, 4).
    """
    Ps = _PAULI_BASIS
    Uc = U.conj().T
    T = torch.einsum('ab,bci,cd,daj->ij', U, Ps, Uc, Ps) / 2
    return torch.real(T)

def unitary_to_transfer_matrix_two_site_truncated(U):
    """
    Generalization to 2-site unitary.

    Parameters:
        U (np.ndarray): 4x4 unitary matrix.

    Returns:
        np.ndarray: Real part of the transfer matrix T of shape (16, 16).
    """
    assert U.shape == (4, 4), "U must be a 4x4 matrix"
    Ps2 = _PAULI_BASIS_2SITE
    Uc = U.conj().T
    T = torch.einsum('a b, b c i, c d, d a j -> i j', U, Ps2, Uc, Ps2) / 4
    return torch.real(T)

def evol_one_site(M, U, site, dir = "l", truncation = False, max_bd = 1024, max_err=1E-10):
    A = M[site].data
    if A.dtype != U.dtype:
        A = A.to(U.dtype)
    A_evol = torch.einsum('l p r, s p -> l s r', A, U)
    M[site].modify(data = A_evol)
    M.left_canonicalize()
    return 1, M


def evol_two_site(M, U, site, dir = "l", truncation = False, max_bd = 1024, max_err=1E-10):
    ## Evolve the pure state by U (not necessarily unitary)
    ## Assuming U is two-site 
    ## site is the first site # of the unitary (1 to N-1)
    U = unitary_to_transfer_matrix_two_site_truncated(U)
    assert site<=M.L-1
    A1 = M[site].data
    A2 = M[site+1].data
    d1 = A1.shape[1]
    d2 = A2.shape[1]
    DL = A1.shape[0]
    DR = A2.shape[2]
    U = torch.reshape(U,(d1,d2,d1,d2))
    if A1.dtype != U.dtype:
        A1 = A1.to(U.dtype)
        A2 = A2.to(U.dtype)
    A_evol = torch.einsum('iab,bcl,jkac->ijkl', A1, A2, U)
    A_evol_mat = A_evol.reshape(DL*d1,DR*d2)
    Q, R = torch.linalg.qr(A_evol_mat)

    if(dir == "l"):
        AL = Q.reshape(DL,d1,len(R))
        AR = R.reshape(len(R), d2, DR)

    M[site].modify(data = AL)
    M[site+1].modify(data = AR)
    return 1, M

def two_site_layer(M, Us, eo = "o", truncation = False, max_bd = 1024, max_err = 1E-10):
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
        norm_site, M = evol_two_site(M, U, site, "l", truncation = truncation, max_bd = max_bd, max_err = max_err)
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

def Pauli_MPS_after_QMLM_output_only(M, T, all_U, p_depolar=0, p_dephaseX=0, p_dephaseY=0, p_dephaseZ=0, max_bd = 1024, max_err = 1E-10, truncation = False, noise_type = "all"):
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
    normS = 1.
    if noise_type == "all":
        for i in range(T):
            Noise_list_depolar = [depolarization_noise_transfer_matrix(p_depolar[i, j]) for j in range(L)]
            Noise_list_dephasingX = [dephasing_noise_transfer_matrix(p_dephaseX[i, j], _DIR_X) for j in range(L)]
            Noise_list_dephasingY = [dephasing_noise_transfer_matrix(p_dephaseY[i, j], _DIR_Y) for j in range(L)]
            Noise_list_dephasingZ = [dephasing_noise_transfer_matrix(p_dephaseZ[i, j], _DIR_Z) for j in range(L)]
            U_list = all_U[i,:L]

            
            norm_i, M = two_site_layer(M, U_list[:L//2], "o", truncation = truncation, max_bd = max_bd, max_err = max_err)
            M.right_canonicalize()

            norm_i, M = two_site_layer(M, U_list[L//2:], "e", truncation = truncation, max_bd = max_bd, max_err = max_err)

            norm_i, M = one_site_layer(M, Noise_list_depolar, max_bd = max_bd, max_err = max_err)

            norm_i, M = one_site_layer(M, Noise_list_dephasingX, max_bd = max_bd, max_err = max_err)

            norm_i, M = one_site_layer(M, Noise_list_dephasingY, max_bd = max_bd, max_err = max_err)

            norm_i, M = one_site_layer(M, Noise_list_dephasingZ, max_bd = max_bd, max_err = max_err)
            M.right_canonicalize()

            if truncation:
                for i, tensor in enumerate(M):
                    inds = list(M[i].inds)
                    inds[1], inds[2] = inds[2], inds[1]
                    M[i].transpose_(*inds)   # reorders axes, does NOT relabel connectivity
                M = qtn.tensor_network_1d_compress(
                    M,
                    method="direct",        # or "dm"
                    max_bond=max_bd,           # hard maximum bond dimension
                    cutoff=max_err,           # error / truncation threshold
                    cutoff_mode="rsum2",    # “discarded weight” style
                    permute_arrays=False,
                )
                thesitetag = qtn.rand_uuid()
                for i, tensor in enumerate(M):
                    if i == 0:
                        new_data = tensor.copy()
                        new_data = tensor.data.reshape(1, tensor.data.shape[0], tensor.data.shape[1])
                        M[i].modify(data=new_data, inds=(thesitetag, tensor.inds[0], tensor.inds[1]))
                    elif i == M.L - 1:
                        new_data = tensor.copy()
                        new_data = tensor.data.reshape(tensor.data.shape[0], 1, tensor.data.shape[1])
                        M[i].modify(data=new_data, inds=(tensor.inds[0], thesitetag, tensor.inds[1]))
                for i, tensor in enumerate(M):
                    inds = list(M[i].inds)
                    inds[1], inds[2] = inds[2], inds[1]
                    M[i].transpose_(*inds)
    elif noise_type == "depolarizing":
        for i in range(T):
            Noise_list_depolar = [depolarization_noise_transfer_matrix(p_depolar[i, j]) for j in range(L)]
            #U_list = [Haar_random_unitary(4) for _ in range(L-1)]
            U_list = all_U[i,:L]

            norm_i, M = two_site_layer(M, U_list[:L//2], "o", truncation = truncation, max_bd = max_bd, max_err = max_err)

            norm_i, M = two_site_layer(M, U_list[L//2:], "e", truncation = truncation, max_bd = max_bd, max_err = max_err)
            norm_i, M = one_site_layer(M, Noise_list_depolar, max_bd = max_bd, max_err = max_err)
            if truncation:
                for i, tensor in enumerate(M):
                    inds = list(M[i].inds)
                    inds[1], inds[2] = inds[2], inds[1]
                    M[i].transpose_(*inds)   # reorders axes, does NOT relabel connectivity
                M = qtn.tensor_network_1d_compress(
                    M,
                    method="direct",        # or "dm"
                    max_bond=max_bd,           # hard maximum bond dimension
                    cutoff=max_err,           # error / truncation threshold
                    cutoff_mode="rsum2",    # “discarded weight” style
                    permute_arrays=False,
                )
                thesitetag = qtn.rand_uuid()
                for i, tensor in enumerate(M):
                    if i == 0:
                        new_data = tensor.copy()
                        new_data = tensor.data.reshape(1, tensor.data.shape[0], tensor.data.shape[1])
                        M[i].modify(data=new_data, inds=(thesitetag, tensor.inds[0], tensor.inds[1]))
                    elif i == M.L - 1:
                        new_data = tensor.copy()
                        new_data = tensor.data.reshape(tensor.data.shape[0], 1, tensor.data.shape[1])
                        M[i].modify(data=new_data, inds=(tensor.inds[0], thesitetag, tensor.inds[1]))
                for i, tensor in enumerate(M):
                    inds = list(M[i].inds)
                    inds[1], inds[2] = inds[2], inds[1]
                    M[i].transpose_(*inds)   # reorders axes, does NOT relabel connectivity
    else:
        raise ValueError(f"noise_type are all, depolarizing. Not: {noise_type}")
    return M

class QMLM_output_only(nn.Module):
    def __init__(self, N, T, param = None, p_depolar = None, p_dephaseX = None, p_dephaseY = None, p_dephaseZ = None, max_bd = 64, max_err = 1E-10):
        super(QMLM_output_only, self).__init__()
        if p_depolar is None:
            p_depolar = torch.zeros((int(T), int(N)), dtype=torch.float64)
        if p_dephaseX is None:
            p_dephaseX = torch.zeros((int(T), int(N)), dtype=torch.float64)
        if p_dephaseY is None:
            p_dephaseY = torch.zeros((int(T), int(N)), dtype=torch.float64)
        if p_dephaseZ is None:
            p_dephaseZ = torch.zeros((int(T), int(N)), dtype=torch.float64)
        self.layers = T
        self.N = N
        if param is None:
            self.params = nn.Parameter(torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64), requires_grad=False)
        else:
            assert param.shape == (T, N, 16), "Parameter list length must match the number of layers T and N."
            self.params = param
        self.p_depolar = torch.nn.Parameter(p_depolar, requires_grad=False)
        self.p_dephaseX = torch.nn.Parameter(p_dephaseX, requires_grad=False)
        self.p_dephaseY = torch.nn.Parameter(p_dephaseY, requires_grad=False)
        self.p_dephaseZ = torch.nn.Parameter(p_dephaseZ, requires_grad=False)
        self.max_bd = max_bd
        self.max_err = max_err
                            

    def forward(self, M_input, truncation = False):
        """ Forward pass of the QMLM model.
        
        Parameters:
            M_input (qtn.MatrixProductState): Input MPS.
        
        Returns:
            qtn.MatrixProductState: Output MPS after applying the QMLM model.
        """
        all_U = construct_all_SU4(self.N, self.layers, self.params)
        M_output = Pauli_MPS_after_QMLM_output_only(M_input, self.layers, all_U, self.p_depolar, self.p_dephaseX, self.p_dephaseY, self.p_dephaseZ, max_bd=self.max_bd, max_err=self.max_err, truncation=truncation, noise_type="all")
        return M_output  # Return the output MPS after applying the QMLM model
    
    def forward_depolarizing_only(self, M_input, truncation = False):
        """ Forward pass of the QMLM model.
        
        Parameters:
            M_input (qtn.MatrixProductState): Input MPS.
        
        Returns:
            qtn.MatrixProductState: Output MPS after applying the QMLM model.
        """
        all_U = construct_all_SU4(self.N, self.layers, self.params)
        M_output = Pauli_MPS_after_QMLM_output_only(M_input, self.layers, all_U, self.p_depolar, max_bd=self.max_bd, max_err=self.max_err, truncation=truncation, noise_type = "depolarizing")
        return M_output  # Return the output MPS after applying the QMLM model
    
    def loss(self, M_input, M_realoutput, i = None):
        """
        Calculate a stable, differentiable loss for the QMLM model using
        the squared Frobenius distance between MPS states.

        Returns:
            torch.Tensor: Scalar loss (requires_grad=True).
        """
        # Forward
        M_output = self.forward(M_input)
        xAA = M_output.H @ M_output              # ⟨A|A⟩
        xBB = M_realoutput.H @ M_realoutput      # ⟨B|B⟩
        xAB = M_output.H @ M_realoutput          # ⟨A|B⟩
        '''if i % 10 == 0:
            print(f"xAA: {xAA}, xBB: {xBB}, xAB: {xAB}")'''
        s = (xAA + xBB - 2.0 * xAB.real)
        return s

    def alt_loss(self, M_input, M_check):
        """ Calculate the loss function for the QMLM model.

        Parameters:
            M_out (qtn.MatrixProductState): Output MPS.
            M_check (qtn.MatrixProductState): Target MPS.

        Returns:
            float: Loss value.
        """
        M_out =  self.forward(M_input)
        return qtn.tensor_network_distance(
            M_out,
            M_check,
            distance_method='auto',
            normalized='squared',
            # contraction options for performance/robustness
        contract_optimize='auto-hq'
    )
    def infidelity(self, M_input, M_realoutput):
        """ Calculate the infidelity for the QMLM model.

        Parameters:
            M_input (qtn.MatrixProductState): Input MPS.
            M_realoutput (qtn.MatrixProductState): Target MPS.

        Returns:
            float: Infidelity value.
        """
        M_output =  self.forward(M_input)
        # Inner products (quimb supports M.H @ N for overlaps)
        xAA = M_output.H @ M_output              # ⟨A|A⟩
        xBB = M_realoutput.H @ M_realoutput      # ⟨B|B⟩
        xAB = M_output.H @ M_realoutput          # ⟨A|B⟩
        return (1.0 - (torch.abs(xAB)**2 / torch.sqrt(xAA * xBB)))


    def infidelity_loss(self, M_input, M_realoutput):
        """ Calculate the loss function for the QMLM model.
        
        Parameters:
            M_input (qtn.MatrixProductState): Input MPS.
            M_realoutput (qtn.MatrixProductState): Real output MPS.
        
        Returns:
            float: Loss value.
        """
        M_output = self.forward(M_input)
        loss_val = tensor_network_distance(M_output, M_realoutput, distance_method='infidelity')
        return loss_val
    
def Pauli_MPS_weight_1(n):
    """
    Generate all Pauli MPS of weight 1 for n sites.
    
    Parameters:
        n (int): Number of sites in the MPS.
    
    Returns:
        list: List of qtn.MatrixProductState objects representing weight-1 Pauli MPS.
    """
    pauli_ops = ['X', 'Y', 'Z']
    MPS_list = []
    for site in range(n):
        for op in pauli_ops:
            M = Identity_init(n, bond_dim=1, phys_dim=4)
            M = operator_assignment_single_site(M, site, op)
            MPS_list.append(M)
    return MPS_list

def get_target_tn(N, T, param = None):
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    MPS_weight1 = Pauli_MPS_weight_1(N)
    QMLM_MPS_output = QMLM_output_only(N, T, param = param)
    target_tn_list = []
    for input in MPS_weight1:
        MPS_target = QMLM_MPS_output.forward(input.copy())
        #target_tn = MPS_target
        for i, tensor in enumerate(MPS_target):
            tensor = tensor.reindex_({f'input{i}': f'k{i}'})
        target_tn = input.copy() | MPS_target
        target_tn_list.append(target_tn)
    
    return target_tn_list, param


def compress_tn_to_mps(tn, N, max_bond=64, cutoff=1e-10):
    """Compress a merged tensor network (e.g. ``mpo | mps``) into a proper MPS.

    This is a convenience utility for the common pattern::

        M_merged = mpo_fit | input_mps   # large uncontracted TN
        M_compressed = compress_tn_to_mps(M_merged, N, max_bond=64)
        loss = tensor_network_distance(M_compressed, target_mps)  # fast

    It uses quimb's 1-D compression (SVD-based direct method) to turn
    the merged TN into an MPS whose bond dimension is at most *max_bond*.

    **Prefer** ``QMLM.forward_compressed`` or ``apply_mpo_to_mps_compressed``
    when possible — they apply the circuit layer-by-layer which is both more
    numerically stable and faster for deep circuits.  This helper is useful as
    a quick drop-in when you already have a merged TN.

    Parameters:
        tn: quimb TensorNetwork (result of ``mpo | mps``).
        N (int): Number of physical sites.
        max_bond (int): Maximum bond dimension.
        cutoff (float): SVD truncation cutoff.

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
        # Fallback: contract everything then do an SVD sweep
        raw = tn.contract(output_inds=outer_inds)
        # raw is a single big tensor; decompose into MPS form
        mps = qtn.MatrixProductState.from_dense(
            raw.data, dims=[4] * N,
            max_bond=max_bond, cutoff=cutoff,
        )
    return mps


def _process_single_mps(args):
    """Worker function for parallel get_input_and_output_MPS.
    Processes a single Pauli MPS through the quantum channel."""
    idx, mps_input, QMLM_MPS_output, noise_type, truncation, N, T, p_depolar = args
    with torch.no_grad():
        if noise_type == "all":
            MPS_target = QMLM_MPS_output.forward(mps_input.copy())
        elif noise_type == "depolarizing":
            MPS_target = QMLM_MPS_output.forward_depolarizing_only(mps_input.copy(), truncation=truncation)
        else:
            raise ValueError(f"noise_type are all, depolarizing. Not: {noise_type}")
        for i, tensor in enumerate(MPS_target):
            tensor.reindex_({f'input{i}': f'k{i}'})
    return idx, MPS_target


def get_input_and_output_MPS(N, T, param, p_depolar, p_dephaseX=None, p_dephaseY=None, p_dephaseZ=None,
                             truncation=False, max_bd=64, max_err=1E-10, noise_type="all",
                             num_threads=None):
    """Generate input weight-1 Pauli MPS and their corresponding output MPS
    after passing through the quantum channel.

    Parameters
    ----------
    num_threads : int or None
        Number of Python threads for the parallel forward passes.
        Defaults to 1 (sequential).  Set > 1 only for standalone use
        outside of ProcessPoolExecutor workers.
    """
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    if p_depolar is None:
        p_depolar = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseX is None:
        p_dephaseX = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseY is None:
        p_dephaseY = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseZ is None:
        p_dephaseZ = torch.zeros((int(T), int(N)), dtype=torch.float64)

    MPS_weight1 = Pauli_MPS_weight_1(N)
    QMLM_MPS_output = QMLM_output_only(
        N, T, param=param, p_depolar=p_depolar,
        p_dephaseX=p_dephaseX, p_dephaseY=p_dephaseY, p_dephaseZ=p_dephaseZ,
        max_bd=max_bd, max_err=max_err,
    )

    total = len(MPS_weight1)
    if num_threads is None:
        num_threads = 1

    if num_threads <= 1:
        # ── Fast sequential path (no thread-pool overhead) ──────────
        target_mps_list = []
        with torch.no_grad():
            for counter, inp in enumerate(MPS_weight1):
                if noise_type == "all":
                    MPS_target = QMLM_MPS_output.forward(inp.copy())
                elif noise_type == "depolarizing":
                    MPS_target = QMLM_MPS_output.forward_depolarizing_only(inp.copy(), truncation=truncation)
                else:
                    raise ValueError(f"noise_type are all, depolarizing. Not: {noise_type}")
                for i, tensor in enumerate(MPS_target):
                    tensor.reindex_({f'input{i}': f'k{i}'})
                target_mps_list.append(MPS_target)
        print(f"N{N}, L{T}, p{p_depolar[0][0]}: processed {total} MPS", flush=True)
    else:
        # ── Threaded path (for standalone use) ──────────────────────
        task_args = [
            (idx, mps, QMLM_MPS_output, noise_type, truncation, N, T, p_depolar)
            for idx, mps in enumerate(MPS_weight1)
        ]
        target_mps_list = [None] * total
        print(f"N{N}, L{T}, p{p_depolar[0][0]}: processing {total} MPS with {num_threads} threads", flush=True)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(_process_single_mps, arg): arg[0] for arg in task_args}
            for future in as_completed(futures):
                idx, MPS_target = future.result()
                target_mps_list[idx] = MPS_target

    return MPS_weight1, target_mps_list, param


def get_random_input_output_MPS(N, T, param, p_depolar, no_sample = 30):
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    mps_list = []
    for i in range(no_sample):
        mps_list.append(random_pauli_MPS(N)[0])
    QMLM_MPS_output = QMLM_output_only(N, T, param = param, p_depolar = p_depolar)
    target_mps_list = []
    for input in mps_list:
        MPS_target = QMLM_MPS_output.forward(input.copy())
        for i, tensor in enumerate(MPS_target):
            tensor = tensor.reindex_({f'input{i}': f'k{i}'})
        target_mps_list.append(MPS_target)
    
    return mps_list, target_mps_list, param

def get_training_data(N, T, param = None, no_sample = 20):
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    mps_list = []
    for i in range(no_sample):
        mps_list.append(random_pauli_MPS(N)[0])
    QMLM_MPS_output = QMLM_output_only(N, T, param = param)
    target_mps_list = []
    for input in mps_list:
        MPS_target = QMLM_MPS_output.forward(input.copy())
        for i, tensor in enumerate(MPS_target):
            tensor = tensor.reindex_({f'input{i}': f'k{i}'})
        target_mps_list.append(MPS_target)
    
    return mps_list, target_mps_list, param

def get_average_loss_weight_tn(tn, tn_target, N, n_times=50, weight = 1):
    total_loss = 0.0
    for _ in range(n_times):
        M_input, weight = random_pauli_MPS(N, weight)
        with torch.no_grad():
            M_realoutput = tn_target | M_input.copy(deep=True)
        M_output = tn | M_input
        M_output.astype_('complex128')
        M_realoutput.astype_('complex128')
        loss = tensor_network_distance(M_output.copy(), M_realoutput.copy())
        total_loss += loss.item()
    average_loss = total_loss / (n_times)
    return average_loss

def ensure_complex_torch(tn, dtype=torch.complex64):
    tn = tn.copy()
    tn.apply_to_arrays(lambda x: torch.as_tensor(x).to(dtype))
    return tn

def Learning_MPO(
    N, MPO_layer, model_to_learn_layer,
    param_list=None, depolarizing_strength=0.2,
    epochs=100, lr=0.01, normalized = False, truncation = False, max_bd=64, max_err=1e-8, noise_type = "all",
    use_compressed=False
):
    times = 3
    if param_list is None:
        param = (
            torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
            + 1j * torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
        )                                               # teacher params [attached_file:1]
    else:
        param = param_list                              # if provided externally [attached_file:1]

    p_depolar_MPO = torch.ones((MPO_layer, N), dtype=torch.float64) * depolarizing_strength*0.5
    p_depolar = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * depolarizing_strength 
    model = QMLM(N, MPO_layer, p_depolar = p_depolar_MPO)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=50, min_lr=0.001)
    learning_loss = []
    clamp_value = min(depolarizing_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)

    MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar, max_bd=max_bd, max_err=max_err, truncation=truncation, noise_type=noise_type
    )
    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)  # freeze input MPS tensors
    for target_tensor in target_mps_list:
        for tensor in target_tensor.tensors:
            tensor.data.requires_grad_(False)

    for epoch in range(epochs):
        losses = []
        if use_compressed:
            # Layer-by-layer application with truncation → output is a proper MPS
            # tensor_network_distance between two MPS is O(N D^3) instead of exponential
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(output_mps_list[sample_idx].astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx] 
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(M_input.astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())
        optimizer.zero_grad()
        loss_this_epoch.backward()                      
        optimizer.step()
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value+0.01)

    for epoch in range(epochs//5):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(output_mps_list[sample_idx].astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx] 
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(M_input.astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())
        optimizer.zero_grad()
        loss_this_epoch.backward()                      
        optimizer.step()
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, min(2*clamp_value+0.01, 1.0))

    # final MPO after training
    final_mpo_fit = model.get_MPO(noise_type=noise_type)                     # rebuild MPO once more if you need it [attached_file:1]
    model_to_learn = QMLM(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar
    )
    mpo_target = model_to_learn.get_MPO()
    testing_loss = 0.0
    for i in range(200):
        print(f"Testing sample {i+1}/200")
        Random_MPS, weight = random_pauli_MPS(N)
        if use_compressed:
            M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type=noise_type)
            for j, tensor in enumerate(M_out):
                tensor.reindex_({f'input{j}': f'k{j}'})
            M_target = mpo_target.astype("complex128") | Random_MPS.astype("complex128")
            loss = tensor_network_distance(M_out.astype("complex128"), M_target.astype("complex128")).item()
        else:
            loss = tensor_network_distance(final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"), mpo_target.astype("complex128") | Random_MPS.astype("complex128")).item()
        testing_loss += loss
    testing_loss /= 200
    print("Testing loss:", testing_loss)
    return model, learning_loss, param, p_depolar, testing_loss




def _compute_single_loss_compressed(sample_idx, output_mps_list, target_mps_list, normalized):
    """Compute loss for one sample (compressed path). Thread-safe."""
    return tensor_network_distance(
        output_mps_list[sample_idx].astype("complex128"),
        target_mps_list[sample_idx].astype("complex128"),
        normalized=normalized,
    )

def _compute_single_loss_mpo(sample_idx, mpo_fit, MPS_weight1, target_mps_list, normalized):
    """Compute loss for one sample (MPO path). Thread-safe."""
    M_input = mpo_fit | MPS_weight1[sample_idx]
    return tensor_network_distance(
        M_input.astype("complex128"),
        target_mps_list[sample_idx].astype("complex128"),
        normalized=normalized,
    )

def _compute_losses_parallel(
    model, MPS_weight1, target_mps_list, normalized,
    use_compressed, noise_type, max_bd, max_err,
    num_threads=1,
):
    """Compute per-sample losses, optionally in parallel using threads.

    Uses ThreadPoolExecutor so the autograd graph is preserved (shared
    memory) and `loss.backward()` works after `torch.stack(losses).mean()`.

    Parameters
    ----------
    num_threads : int
        Number of threads.  1 = sequential (no pool overhead).
    """
    n_samples = len(MPS_weight1)

    if use_compressed:
        output_mps_list = apply_mpo_to_mps_compressed(
            model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
        )
        if num_threads <= 1:
            losses = [
                _compute_single_loss_compressed(i, output_mps_list, target_mps_list, normalized)
                for i in range(n_samples)
            ]
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = {
                    pool.submit(_compute_single_loss_compressed, i, output_mps_list, target_mps_list, normalized): i
                    for i in range(n_samples)
                }
                losses = [None] * n_samples
                for future in as_completed(futures):
                    idx = futures[future]
                    losses[idx] = future.result()
    else:
        mpo_fit = model.get_MPO(noise_type=noise_type)
        if num_threads <= 1:
            losses = [
                _compute_single_loss_mpo(i, mpo_fit, MPS_weight1, target_mps_list, normalized)
                for i in range(n_samples)
            ]
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = {
                    pool.submit(_compute_single_loss_mpo, i, mpo_fit, MPS_weight1, target_mps_list, normalized): i
                    for i in range(n_samples)
                }
                losses = [None] * n_samples
                for future in as_completed(futures):
                    idx = futures[future]
                    losses[idx] = future.result()

    return losses


def Learning_MPO_scheduler(
    N, MPO_layer, model_to_learn_layer,
    param_list=None, depolarizing_strength=0.2,
    epochs=100, lr=0.01, normalized = False, max_bd=32, max_err=1E-6, truncation = False, noise_type = "all",
    use_compressed=False, num_threads=None
):
    if param_list is None:
        param = (
            torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
            + 1j * torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
        )
    else:
        param = param_list

    p_depolar_MPO = torch.ones((MPO_layer, N), dtype=torch.float64) * depolarizing_strength * 0.5
    p_depolar = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * depolarizing_strength
    model = QMLM(N, MPO_layer, p_depolar=p_depolar_MPO)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # we want to reduce LR when loss stops decreasing
        factor=0.5,      # shrink LR by a factor of 2
        patience=10,     # epochs to wait with no improvement
        threshold=5e-4,
        min_lr=1e-5,     # do not go below this LR
    )

    learning_loss = []
    clamp_value = min(depolarizing_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)

    MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar, max_bd=max_bd, max_err=max_err,truncation=truncation, noise_type="depolarizing", num_threads=num_threads
    )
    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)
    for target_tensor in target_mps_list:
        for tensor in target_tensor.tensors:
            tensor.data.requires_grad_(False)

    # main training loop
    for epoch in range(epochs):
        #print(f"Epoch {epoch+1}/{epochs}")
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    output_mps_list[sample_idx].astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad()
        loss_this_epoch.backward()
        optimizer.step()
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value + 0.01)

        # step scheduler with the current epoch loss (as metric)
        scheduler.step(loss_this_epoch.detach())
        if epoch % 1 == 0:
            print(f"N{N}, T{MPO_layer}, L{model_to_learn_layer}, p{depolarizing_strength}, Epoch {epoch}, Loss:", loss_this_epoch.item(), end = ", ")
            print(f"Learning rate at epoch {epoch}:", optimizer.param_groups[0]['lr'])

    # fine-tuning loop
    for epoch in range(epochs // 5):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    output_mps_list[sample_idx].astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        #print(f"Loss at fine-tune epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad()
        loss_this_epoch.backward()
        optimizer.step()
        model.p_depolar.data = torch.clamp(
            model.p_depolar.data,
            0.0,
            min(2 * clamp_value + 0.01, 1.0)
        )

        # also update scheduler here (still monitoring the same loss)
        scheduler.step(loss_this_epoch.detach())
        if epoch % 1 == 0:
            print(f"Fine-tune, N{N}, T{MPO_layer}, L{model_to_learn_layer}, p{depolarizing_strength}, Epoch {epoch}, Loss:", loss_this_epoch.item(), end = ", ")
            print(f"Learning rate at epoch {epoch}:", optimizer.param_groups[0]['lr'])

    # final MPO after training
    final_mpo_fit = model.get_MPO(noise_type=noise_type)
    testing_loss_list = []
    QMLM_MPS_output = QMLM_output_only(N, model_to_learn_layer, param = param, p_depolar = p_depolar, max_bd = max_bd, max_err = max_err)
    num_samples = 100

    def _test_single_sample(j):
        """Evaluate one random test sample. No gradients needed."""
        with torch.no_grad():
            Random_MPS, weight = random_pauli_MPS(N)
            MPS_target = QMLM_MPS_output.forward_depolarizing_only(Random_MPS.copy(), truncation=truncation)
            for i, tensor in enumerate(MPS_target):
                tensor = tensor.reindex_({f'input{i}': f'k{i}'})
            if use_compressed:
                M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type=noise_type)
                for i, tensor in enumerate(M_out):
                    tensor.reindex_({f'input{i}': f'k{i}'})
                loss = tensor_network_distance(
                    M_out.astype("complex128"),
                    MPS_target.astype("complex128")
                ).item()
            else:
                loss = tensor_network_distance(
                    final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"),
                    MPS_target.astype("complex128")
                ).item()
        return loss

    test_threads = max(1, num_threads if num_threads else 1)
    if test_threads <= 1:
        testing_loss_list = [_test_single_sample(j) for j in range(num_samples)]
    else:
        with ThreadPoolExecutor(max_workers=test_threads) as pool:
            futures = {pool.submit(_test_single_sample, j): j for j in range(num_samples)}
            testing_loss_list = [None] * num_samples
            for future in as_completed(futures):
                idx = futures[future]
                testing_loss_list[idx] = future.result()

    testing_loss = sum(testing_loss_list) / num_samples
    print(f"N{N}, T{MPO_layer}, L{model_to_learn_layer}, p{depolarizing_strength}, Testing loss:", testing_loss)
    return model, learning_loss, param, p_depolar, testing_loss, testing_loss_list, model.params.detach().numpy(), model.p_depolar.detach().numpy(), model.p_dephaseX.detach().numpy(), model.p_dephaseY.detach().numpy(), model.p_dephaseZ.detach().numpy()


def Learning_MPO_dephasing_noise_only(
    N, MPO_layer, model_to_learn_layer,
    param_list=None, dephasingX_strength=0.2, dephasingY_strength=0.2, dephasingZ_strength=0.2,
    epochs=100, lr=0.01, normalized = False, max_bd=64, max_err=1e-8,
    use_compressed=False
):
    times = 3
    if param_list is None:
        param = (
            torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
            + 1j * torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
        )                                               # teacher params [attached_file:1]
    else:
        param = param_list                              # if provided externally [attached_file:1]

    p_depolar_MPO = torch.zeros((MPO_layer, N), dtype=torch.float64)
    p_depolar = torch.zeros((model_to_learn_layer, N), dtype=torch.float64) 
    p_dephaseX = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingX_strength
    p_dephaseY = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingY_strength
    p_dephaseZ = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingZ_strength
    model = QMLM(N, MPO_layer)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=50, min_lr=0.001)
    learning_loss = []
    clamp_value_X = min(dephasingX_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
    clamp_value_Y = min(dephasingY_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
    clamp_value_Z = min(dephasingZ_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)

    MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar, p_dephaseX = p_dephaseX, p_dephaseY = p_dephaseY, p_dephaseZ = p_dephaseZ
    )
    for MPS_weight1_tensor in MPS_weight1:
        #MPS_weight1_tensor = MPS_weight1_tensor.astype("complex64")
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)  # freeze input MPS tensors


    for epoch in range(epochs):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type="dephasing"
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(output_mps_list[sample_idx].astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO_without_depolarizing_noise()
            #indices_this_epoch = np.random.permutation(np.arange(len(MPS_weight1)))[:min(int(len(MPS_weight1)*5*epoch/epochs)+1, len(MPS_weight1))]
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx] 
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(M_input.astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())
        optimizer.zero_grad()
        loss_this_epoch.backward()
        optimizer.step()
        model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, clamp_value_X+1e-6)
        model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, clamp_value_Y+1e-6)
        model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, clamp_value_Z+1e-6)

    for epoch in range(epochs//5):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type="dephasing"
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(output_mps_list[sample_idx].astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO_without_depolarizing_noise()
            #indices_this_epoch = np.random.permutation(np.arange(len(MPS_weight1)))[:min(int(len(MPS_weight1)*5*epoch/epochs)+1, len(MPS_weight1))]
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx] 
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(M_input.astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())
        optimizer.zero_grad()
        loss_this_epoch.backward()                      
        optimizer.step()
        model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, min(2*clamp_value_X+1e-6, 1.0))
        model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, min(2*clamp_value_Y+1e-6, 1.0))
        model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, min(2*clamp_value_Z+1e-6, 1.0))
    #scheduler.step(loss_this_epoch.item())
    #optimizer = optim.Adam(model.parameters(), lr=lr)


    # final MPO after training
    final_mpo_fit = model.get_MPO()                     # rebuild MPO once more if you need it [attached_file:1]
    model_to_learn = QMLM(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar, p_dephaseX = p_dephaseX, p_dephaseY = p_dephaseY, p_dephaseZ = p_dephaseZ
    )
    mpo_target = model_to_learn.get_MPO()
    testing_loss_list = []
    testing_loss = 0.0
    for i in range(200):
        Random_MPS, weight = random_pauli_MPS(N)
        if use_compressed:
            M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type="all")
            for j, tensor in enumerate(M_out):
                tensor.reindex_({f'input{j}': f'k{j}'})
            M_target = mpo_target.astype("complex128") | Random_MPS.astype("complex128")
            loss = tensor_network_distance(M_out.astype("complex128"), M_target.astype("complex128")).item()
        else:
            loss = tensor_network_distance(final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"), mpo_target.astype("complex128") | Random_MPS.astype("complex128")).item()
        testing_loss_list.append(loss)
        testing_loss += loss
    testing_loss /= 200
    print("Testing loss:", testing_loss)
    return model, learning_loss, param, p_depolar, testing_loss, testing_loss_list, model.params, model.p_depolar, model.p_dephaseX, model.p_dephaseY, model.p_dephaseZ



def Learning_MPO_dephasing_noise_only_scheduler(
    N, MPO_layer, model_to_learn_layer,
    param_list=None, dephasingX_strength=0.2, dephasingY_strength=0.2, dephasingZ_strength=0.2,
    epochs=100, lr=0.01, normalized=False, max_bd=64, max_err=1e-8,
    use_compressed=False
):
    times = 3
    if param_list is None:
        param = (
            torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
            + 1j * torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
        )
    else:
        param = param_list

    p_depolar_MPO = torch.zeros((MPO_layer, N), dtype=torch.float64)
    p_depolar = torch.zeros((model_to_learn_layer, N), dtype=torch.float64)
    p_dephaseX = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingX_strength
    p_dephaseY = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingY_strength
    p_dephaseZ = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingZ_strength

    model = QMLM(N, MPO_layer)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)

    # Add ReduceLROnPlateau scheduler (monitoring the training loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,        # shrink LR by 2x when plateauing
        patience=10,       # epochs with no improvement before reducing LR
        threshold=5e-4,
        min_lr=1e-7,
    )

    learning_loss = []
    clamp_value_X = min(dephasingX_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)
    clamp_value_Y = min(dephasingY_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)
    clamp_value_Z = min(dephasingZ_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)

    MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
        N, model_to_learn_layer, param=param,
        p_depolar=p_depolar,
        p_dephaseX=p_dephaseX, p_dephaseY=p_dephaseY, p_dephaseZ=p_dephaseZ
    )
    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)

    # ---------- first training phase ----------
    for epoch in range(epochs):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type="dephasing"
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    output_mps_list[sample_idx].astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized,
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO_without_depolarizing_noise()
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized,
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad()
        loss_this_epoch.backward()
        optimizer.step()

        # clamp noise parameters
        model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, clamp_value_X + 1e-6)
        model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, clamp_value_Y + 1e-6)
        model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, clamp_value_Z + 1e-6)

        # update LR based on epoch loss
        scheduler.step(loss_this_epoch)
        if epoch % 10 == 0:
            print(f"Learning Rate at epoch {epoch}: {optimizer.param_groups[0]['lr']}")

    # ---------- second training phase ----------
    for epoch in range(epochs // 5):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type="dephasing"
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    output_mps_list[sample_idx].astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized,
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO_without_depolarizing_noise()
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized,
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss (phase 2) at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad()
        loss_this_epoch.backward()
        optimizer.step()

        model.p_dephaseX.data = torch.clamp(
            model.p_dephaseX.data, 0.0, min(2 * clamp_value_X + 1e-6, 1.0)
        )
        model.p_dephaseY.data = torch.clamp(
            model.p_dephaseY.data, 0.0, min(2 * clamp_value_Y + 1e-6, 1.0)
        )
        model.p_dephaseZ.data = torch.clamp(
            model.p_dephaseZ.data, 0.0, min(2 * clamp_value_Z + 1e-6, 1.0)
        )

        # keep scheduling in phase 2 as well
        scheduler.step(loss_this_epoch)
        if epoch % 10 == 0:
            print(f"Learning Rate at epoch {epoch}: {optimizer.param_groups[0]['lr']}")

    # final MPO after training
    final_mpo_fit = model.get_MPO()
    model_to_learn = QMLM(
        N, model_to_learn_layer,
        param=param,
        p_depolar=p_depolar,
        p_dephaseX=p_dephaseX, p_dephaseY=p_dephaseY, p_dephaseZ=p_dephaseZ
    )
    mpo_target = model_to_learn.get_MPO()
    testing_loss = 0.0
    for i in range(200):
        Random_MPS, weight = random_pauli_MPS(N)
        if use_compressed:
            M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type="all")
            for j, tensor in enumerate(M_out):
                tensor.reindex_({f'input{j}': f'k{j}'})
            M_target = mpo_target.astype("complex128") | Random_MPS.astype("complex128")
            loss = tensor_network_distance(M_out.astype("complex128"), M_target.astype("complex128")).item()
        else:
            loss = tensor_network_distance(
                final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"),
                mpo_target.astype("complex128") | Random_MPS.astype("complex128"),
            ).item()
        testing_loss += loss
    testing_loss /= 200
    print("Testing loss:", testing_loss)
    return model, learning_loss, param, p_depolar, testing_loss


def get_OSE_output(N, T, param, p_depolar, p_dephaseX=None, p_dephaseY=None, p_dephaseZ=None, truncation = False, max_bd = 64, max_err = 1E-10, noise_type="all"):
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    if p_depolar is None:
        p_depolar = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseX is None:
        p_dephaseX = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseY is None:
        p_dephaseY = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseZ is None:
        p_dephaseZ = torch.zeros((int(T), int(N)), dtype=torch.float64)
    MPS_weight1 = Pauli_MPS_weight_1(N)
    MPS_OSE_in = MPS_weight1[0]
    for i in range(1, len(MPS_weight1)):
        MPS_OSE_in = MPS_OSE_in + MPS_weight1[i]
    MPS_OSE_in = MPS_OSE_in/len(MPS_weight1)
    QMLM_MPS_output = QMLM_output_only(N, T, param = param, p_depolar = p_depolar, p_dephaseX = p_dephaseX, p_dephaseY = p_dephaseY, p_dephaseZ = p_dephaseZ, max_bd = max_bd, max_err = max_err)
    with torch.no_grad():
        if noise_type == "all":
            MPS_target = QMLM_MPS_output.forward(MPS_OSE_in.copy())
        elif noise_type == "depolarizing":
            MPS_target = QMLM_MPS_output.forward_depolarizing_only(MPS_OSE_in.copy(), truncation=truncation)
        else:
            raise ValueError(f"noise_type are all, depolarizing. Not: {noise_type}")
        for i, tensor in enumerate(MPS_target):
            tensor = tensor.reindex_({f'input{i}': f'k{i}'})
    #MPS_target = QMLM_MPS_output.forward_depolarizing_only(MPS_OSE_in.copy(), truncation=truncation)
    return MPS_target, param

def TDME_two_site_hamiltonian(l: int, N: int, mu: float, J: float, t: float = 1.0) -> np.ndarray:
    """
    Return the local 2-site Hamiltonian h_{l, l+1}  (4×4 complex matrix).

    Full Hamiltonian:
        H = -J Σ_{l=1}^{n-1} (a†_l a_{l+1} + h.c.)
            - μ(t) (Σ_{l=1}^{n/2} n_l  -  Σ_{l=n/2+1}^{n} n_l)

    Decomposition:  H = Σ_{l=1}^{n-1} h_{l, l+1}

    On-site chemical-potential terms are split equally between the two
    bonds sharing each interior site.  End sites (l=1, l=n) belong to
    exactly one bond and receive full weight.

    Parameters
    ----------
    l   : int   Bond index (1-based), 1 ≤ l ≤ n-1.
    n   : int   Total number of sites (must be even).
    J   : float Hopping amplitude.
    mu  : float Value of μ(t) at the current time step.

    Returns
    -------
    np.ndarray  4×4 complex Hermitian matrix in the two-site basis
                { |00⟩, |01⟩, |10⟩, |11⟩ }.
    """
    # ── Single-site operators (basis: |0⟩ = empty, |1⟩ = occupied) ─────────────
    mu_t = mu(t) if callable(mu) else float(mu)
    I2   = np.eye(2, dtype=complex)
    a    = np.array([[0, 1], [0, 0]], dtype=complex)   # annihilation: |0⟩⟨1|
    adag = np.array([[0, 0], [1, 0]], dtype=complex)   # creation:     |1⟩⟨0|
    num  = adag @ a                                    # number:       |1⟩⟨1|
    if N % 2 != 0:
        raise ValueError(f"N must be even; got N={N}.")
    if not (0 <= l <= N - 1):
        raise ValueError(f"l must satisfy 0 ≤ l ≤ N-1={N-1}; got l={l}.")

    # ── Hopping: -J (a†_l ⊗ a_{l+1} + h.c.) ────────────────────────────────
    H = -J * (np.kron(adag, a) + np.kron(a, adag))

    # ── Chemical potential ────────────────────────────────────────────────────
    # sign: +1 for left half (sites 1..n/2), -1 for right half
    def sign(site: int) -> float:
        return 1.0 if site <= N // 2 else -1.0

    # Boundary sites get full weight; interior sites are shared → half weight
    w_l   = 1.0 if l == 0       else 0.5
    w_lp1 = 1.0 if (l + 1) == N else 0.5

    H += -mu_t * sign(l)     * w_l   * np.kron(num, I2)
    H += -mu_t * sign(l + 1) * w_lp1 * np.kron(I2, num)

    return H

def construct_TDME_unitary(N: int, T:int, r: float, mu: float, J: float):
    """
    Construct the two-site unitary U = exp(-i H dt) for time evolution.

    Parameters
    ----------
    H   : np.ndarray  4×4 complex Hermitian matrix (two-site Hamiltonian).
    dt  : float       Time step size.

    Returns
    -------
    np.ndarray  4×4 complex unitary matrix representing the time evolution operator.
    """
    from scipy.linalg import expm
    all_unitary = []
    for layer in range(T):
        all_unitary_T = []
        for i in range(0, N, 2):
            H = TDME_two_site_hamiltonian(i, N, mu=mu, J=J, t=r*layer)
            evolu = expm(-1j * H * r)
            if not isinstance(evolu, torch.Tensor):
                evolu = torch.from_numpy(evolu)
            all_unitary_T.append(evolu)
        for i in range(1, N, 2):
            H = TDME_two_site_hamiltonian(i, N, mu=mu, J=J, t=r*layer)
            evolu = expm(-1j * H * r)
            if not isinstance(evolu, torch.Tensor):
                evolu = torch.from_numpy(evolu)
            all_unitary_T.append(evolu)
        all_unitary.append(all_unitary_T)
    return all_unitary

def construct_jump_matrices(N: int, gamma, r: float) -> list:
    jump_matrices = []
    for i in range(N):
        beta = np.exp(-gamma[i] * r / 2)
        jump_matrix = np.diag([1, beta, beta, 1])
        if not isinstance(jump_matrix, torch.Tensor):
            jump_matrix = torch.from_numpy(jump_matrix)
        jump_matrices.append(jump_matrix)
    return jump_matrices

def sanitize_mps(M):
    for tensor in M:
        data = tensor.data
        if not np.all(np.isfinite(data)):
            data[~np.isfinite(data)] = 0.0
        norm = np.linalg.norm(data)
        if norm > 0:
            tensor.modify(data=data / norm)
    return M

def Pauli_MPS_after_TDME_output_only(M, T, r, all_unitary, all_jumping, max_bd = 1024, max_err = 1E-10, truncation = False):
    """ r is t/T, the total evolution time divided into T layers, and we want the output at time t = r*T. """
    L = M.L

    for i in range(T):
        norm_i, M = two_site_layer(M, all_unitary[i][:L//2], "o", truncation = truncation, max_bd = max_bd, max_err = max_err)
        M.right_canonicalize()

        norm_i, M = two_site_layer(M, all_unitary[i][L//2:], "e", truncation = truncation, max_bd = max_bd, max_err = max_err)

        norm_i, M = one_site_layer(M, all_jumping, truncation = truncation, max_bd = max_bd, max_err = max_err)
        M.right_canonicalize()

        if truncation:
            for i, tensor in enumerate(M):
                inds = list(M[i].inds)
                inds[1], inds[2] = inds[2], inds[1]
                M[i].transpose_(*inds)   # reorders axes, does NOT relabel connectivity
            try:
                M = qtn.tensor_network_1d_compress(
                    M,
                    method="direct",        # or "dm"
                    max_bond=max_bd,           # hard maximum bond dimension
                    cutoff=max_err,           # error / truncation threshold
                    cutoff_mode="rsum2",    # “discarded weight” style
                    permute_arrays=False,
                )
            except:
                print("Compression failed at layer", i, "with max_bd =", max_bd, "and max_err =", max_err)
            thesitetag = qtn.rand_uuid()
            for i, tensor in enumerate(M):
                if i == 0:
                    new_data = tensor.copy()
                    new_data = tensor.data.reshape(1, tensor.data.shape[0], tensor.data.shape[1])
                    M[i].modify(data=new_data, inds=(thesitetag, tensor.inds[0], tensor.inds[1]))
                elif i == M.L - 1:
                    new_data = tensor.copy()
                    new_data = tensor.data.reshape(tensor.data.shape[0], 1, tensor.data.shape[1])
                    M[i].modify(data=new_data, inds=(tensor.inds[0], thesitetag, tensor.inds[1]))
            for i, tensor in enumerate(M):
                inds = list(M[i].inds)
                inds[1], inds[2] = inds[2], inds[1]
                M[i].transpose_(*inds)
    return M
class TDME(nn.Module):
    def __init__(self, N, T, mu, gamma, J=1, max_bd = 64, max_err = 1E-10):
        """
        N: int
        T: int
        mu: float
        gamma: list of float, length N
        J: float
        max_bd: int
        max_err: float
        """
        super(TDME, self).__init__()
        self.T = T
        self.N = N
        self.max_bd = max_bd
        self.max_err = max_err
        self.mu = mu
        self.J = J
        self.gamma = gamma

    def forward(self, M_input, t = 1, truncation = False):
        """ Forward pass of the QMLM model.
        
        Parameters:
            M_input (qtn.MatrixProductState): Input MPS.
        
        Returns:
            qtn.MatrixProductState: Output MPS after applying the QMLM model.
        """
        r = t/self.T
        all_unitary = construct_TDME_unitary(self.N, self.T, r=r, mu=self.mu, J=self.J)
        all_jumping = construct_jump_matrices(self.N, self.gamma, r=r)
        M_output, _ = Pauli_MPS_after_TDME_output_only(M_input, self.T, r=r, all_unitary=all_unitary, all_jumping=all_jumping, max_bd=self.max_bd, max_err=self.max_err, truncation=truncation)
        return M_output


def _process_single_mps_tdme(args):
    """Worker for parallel get_input_and_output_MPS_TDME.
    Receives pre-computed unitaries and jump matrices to avoid redundant
    scipy.linalg.expm calls across samples."""
    idx, mps_input, T, r, all_unitary, all_jumping, max_bd, max_err, truncation = args
    with torch.no_grad():
        try:
            MPS_target, truncation_error = Pauli_MPS_after_TDME_output_only(
                mps_input.copy(), T, r=r,
                all_unitary=all_unitary, all_jumping=all_jumping,
                max_bd=max_bd, max_err=max_err, truncation=truncation,
            )
            for i, tensor in enumerate(MPS_target):
                tensor.reindex_({f'input{i}': f'k{i}'})
            return idx, MPS_target, False   # (index, result, skipped)
        except Exception:
            return idx, None, True          # (index, None, skipped)


def get_input_and_output_MPS_TDME(N, T, mu, gamma, J=1, t=1,
                                  truncation=False, max_bd=64, max_err=1E-10,
                                  num_threads=None):
    """Generate input weight-1 Pauli MPS and their TDME outputs.

    Parameters
    ----------
    num_threads : int or None
        Number of Python threads for parallel forward passes.
        Defaults to 1 (sequential).  Set > 1 only for standalone use
        outside of ProcessPoolExecutor workers.
    """
    MPS_weight1 = Pauli_MPS_weight_1(N)
    total = len(MPS_weight1)

    # Pre-compute unitaries and jump matrices ONCE (they are identical
    # for every input MPS — they depend only on model parameters).
    r = t / T
    all_unitary = construct_TDME_unitary(N, T, r=r, mu=mu, J=J)
    all_jumping = construct_jump_matrices(N, gamma, r=r)

    if num_threads is None:
        num_threads = 1

    if num_threads <= 1:
        # ── Fast sequential path (no thread-pool overhead) ──────────
        target_mps_list = []
        skipped_MPS = []
        with torch.no_grad():
            for counter, inp in enumerate(MPS_weight1):
                print(f"Processing MPS {counter+1}/{total}", flush=True)
                try:
                    MPS_target, _ = Pauli_MPS_after_TDME_output_only(
                        inp.copy(), T, r=r,
                        all_unitary=all_unitary, all_jumping=all_jumping,
                        max_bd=max_bd, max_err=max_err, truncation=truncation,
                    )
                    for i, tensor in enumerate(MPS_target):
                        tensor.reindex_({f'input{i}': f'k{i}'})
                    target_mps_list.append(MPS_target)
                except Exception:
                    print(f"Failed to process MPS {counter+1} due to truncation, skipping...")
                    skipped_MPS.append(inp)
        MPS_weight1 = [mps for mps in MPS_weight1 if mps not in skipped_MPS]
        print(len(MPS_weight1), len(target_mps_list), flush=True)
    else:
        # ── Threaded path ───────────────────────────────────────────
        task_args = [
            (idx, mps, T, r, all_unitary, all_jumping, max_bd, max_err, truncation)
            for idx, mps in enumerate(MPS_weight1)
        ]
        results = [None] * total
        skipped_indices = set()
        print(f"Processing {total} MPS with {num_threads} threads", flush=True)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(_process_single_mps_tdme, arg): arg[0]
                for arg in task_args
            }
            for future in as_completed(futures):
                idx, MPS_target, skipped = future.result()
                if skipped:
                    print(f"Failed to process MPS {idx+1} due to truncation, skipping...")
                    skipped_indices.add(idx)
                else:
                    results[idx] = MPS_target

        # Filter out skipped entries, preserving order
        target_mps_list = [r for i, r in enumerate(results) if i not in skipped_indices]
        MPS_weight1 = [m for i, m in enumerate(MPS_weight1) if i not in skipped_indices]
        print(len(MPS_weight1), len(target_mps_list), flush=True)

    return MPS_weight1, target_mps_list


def Learning_TDME_scheduler(
    N, MPO_layer, model_to_learn_layer, mu, gamma, J=1, t = 1,
    epochs=100, lr=0.01, normalized = False, max_bd=128, max_err=1E-8, truncation = False, noise_type = "all", use_scheduler = True,
    use_compressed=True, num_threads=None
):
    model = QMLM(N, MPO_layer)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # we want to reduce LR when loss stops decreasing
        factor=0.5,      # shrink LR by a factor of 2
        patience=10,     # epochs to wait with no improvement
        threshold=5e-4,
        min_lr=1e-5,     # do not go below this LR
    )
    depolarizing_strength = gamma[0]
    learning_loss = []
    clamp_value = min(depolarizing_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)

    MPS_weight1, target_mps_list = get_input_and_output_MPS_TDME(
        N, model_to_learn_layer, mu=mu, gamma=gamma, J=J, t=t,
        max_bd=max_bd, max_err=max_err, truncation=truncation,
        num_threads=num_threads,
    )
    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)
    for target_tensor in target_mps_list:
        for tensor in target_tensor.tensors:
            tensor.data.requires_grad_(False)

    # main training loop
    if noise_type == "depolarizing":
        for epoch in range(epochs):
            losses = []
            if use_compressed:
                output_mps_list = apply_mpo_to_mps_compressed(
                    model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
                )
                for sample_idx in range(len(MPS_weight1)):
                    M_check = target_mps_list[sample_idx]
                    loss = tensor_network_distance(
                        output_mps_list[sample_idx].astype("complex128"),
                        M_check.astype("complex128"),
                        normalized=normalized
                    )
                    losses.append(loss)
            else:
                mpo_fit = model.get_MPO(noise_type=noise_type)
                for sample_idx in range(len(MPS_weight1)):
                    M_input = mpo_fit | MPS_weight1[sample_idx]
                    M_check = target_mps_list[sample_idx]
                    loss = tensor_network_distance(
                        M_input.astype("complex128"),
                        M_check.astype("complex128"),
                        normalized=normalized
                    )
                    losses.append(loss)

            loss_this_epoch = torch.stack(losses).mean()
            learning_loss.append(loss_this_epoch.item())

            optimizer.zero_grad()
            loss_this_epoch.backward()
            optimizer.step()
            model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value + 0.01)

            # step scheduler with the current epoch loss (as metric)
            if use_scheduler:
                scheduler.step(loss_this_epoch.detach())
            if epoch % 10 == 0:
                print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
                print(f"Learning rate at epoch {epoch}:", optimizer.param_groups[0]['lr'])

        # fine-tuning loop
        for epoch in range(epochs // 4):
            losses = []
            if use_compressed:
                output_mps_list = apply_mpo_to_mps_compressed(
                    model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
                )
                for sample_idx in range(len(MPS_weight1)):
                    M_check = target_mps_list[sample_idx]
                    loss = tensor_network_distance(
                        output_mps_list[sample_idx].astype("complex128"),
                        M_check.astype("complex128"),
                        normalized=normalized
                    )
                    losses.append(loss)
            else:
                mpo_fit = model.get_MPO(noise_type=noise_type)
                for sample_idx in range(len(MPS_weight1)):
                    M_input = mpo_fit | MPS_weight1[sample_idx]
                    M_check = target_mps_list[sample_idx]
                    loss = tensor_network_distance(
                        M_input.astype("complex128"),
                        M_check.astype("complex128"),
                        normalized=normalized
                    )
                    losses.append(loss)

            loss_this_epoch = torch.stack(losses).mean()
            print(f"Loss at fine-tune epoch {epoch}:", loss_this_epoch.item())
            learning_loss.append(loss_this_epoch.item())

            optimizer.zero_grad()
            loss_this_epoch.backward()
            optimizer.step()
            model.p_depolar.data = torch.clamp(
                model.p_depolar.data,
                0.0,
                min(2 * clamp_value + 0.01, 1.0)
            )
    elif noise_type == "dephasing":
        dephasingX_strength = gamma[0]*2
        dephasingY_strength = gamma[0]*2
        dephasingZ_strength = gamma[0]*2
        clamp_value_X = min(dephasingX_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
        clamp_value_Y = min(dephasingY_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
        clamp_value_Z = min(dephasingZ_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
        for epoch in range(epochs):
            losses = []
            if use_compressed:
                output_mps_list = apply_mpo_to_mps_compressed(
                    model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
                )
                for sample_idx in range(len(MPS_weight1)):
                    M_check = target_mps_list[sample_idx]
                    loss = tensor_network_distance(
                        output_mps_list[sample_idx].astype("complex128"),
                        M_check.astype("complex128"),
                        normalized=normalized
                    )
                    losses.append(loss)
            else:
                mpo_fit = model.get_MPO(noise_type=noise_type)
                for sample_idx in range(len(MPS_weight1)):
                    M_input = mpo_fit | MPS_weight1[sample_idx]
                    M_check = target_mps_list[sample_idx]
                    loss = tensor_network_distance(
                        M_input.astype("complex128"),
                        M_check.astype("complex128"),
                        normalized=normalized
                    )
                    losses.append(loss)

            loss_this_epoch = torch.stack(losses).mean()
            learning_loss.append(loss_this_epoch.item())

            optimizer.zero_grad()
            loss_this_epoch.backward()
            optimizer.step()
            model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, clamp_value_X+1e-6)
            model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, clamp_value_Y+1e-6)
            model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, clamp_value_Z+1e-6)

            # step scheduler with the current epoch loss (as metric)
            if use_scheduler:
                scheduler.step(loss_this_epoch.detach())
            if epoch % 10 == 0:
                print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
                print(f"Learning rate at epoch {epoch}:", optimizer.param_groups[0]['lr'])

        # fine-tuning loop
        for epoch in range(epochs // 5):
            losses = []
            if use_compressed:
                output_mps_list = apply_mpo_to_mps_compressed(
                    model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
                )
                for sample_idx in range(len(MPS_weight1)):
                    M_check = target_mps_list[sample_idx]
                    loss = tensor_network_distance(
                        output_mps_list[sample_idx].astype("complex128"),
                        M_check.astype("complex128"),
                        normalized=normalized
                    )
                    losses.append(loss)
            else:
                mpo_fit = model.get_MPO(noise_type=noise_type)
                for sample_idx in range(len(MPS_weight1)):
                    M_input = mpo_fit | MPS_weight1[sample_idx]
                    M_check = target_mps_list[sample_idx]
                    loss = tensor_network_distance(
                        M_input.astype("complex128"),
                        M_check.astype("complex128"),
                        normalized=normalized
                    )
                    losses.append(loss)

            loss_this_epoch = torch.stack(losses).mean()
            print(f"Loss at fine-tune epoch {epoch}:", loss_this_epoch.item())
            learning_loss.append(loss_this_epoch.item())

            optimizer.zero_grad()
            loss_this_epoch.backward()
            optimizer.step()
            model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, min(2*clamp_value_X+1e-6, 1.0))
            model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, min(2*clamp_value_Y+1e-6, 1.0))
            model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, min(2*clamp_value_Z+1e-6, 1.0))

        # also update scheduler here (still monitoring the same loss)
        if use_scheduler:
            scheduler.step(loss_this_epoch.detach())
        if epoch % 10 == 0:
            print(f"Learning rate at fine-tune epoch {epoch}:", optimizer.param_groups[0]['lr'])

    # final MPO after training
    final_mpo_fit = model.get_MPO(noise_type=noise_type)
    num_samples = 200

    # Pre-compute target-model unitaries and jump matrices ONCE
    # (identical for every test sample — avoids N×T expm calls per sample).
    r_target = t / model_to_learn_layer
    target_all_unitary = construct_TDME_unitary(N, model_to_learn_layer, r=r_target, mu=mu, J=J)
    target_all_jumping = construct_jump_matrices(N, gamma, r=r_target)

    def _test_single_sample_tdme(j):
        """Evaluate one random test sample. No gradients needed."""
        with torch.no_grad():
            Random_MPS, weight = random_pauli_MPS(N)
            try:
                MPS_target, _ = Pauli_MPS_after_TDME_output_only(
                    Random_MPS.copy(), model_to_learn_layer, r=r_target,
                    all_unitary=target_all_unitary,
                    all_jumping=target_all_jumping,
                    max_bd=max_bd, max_err=max_err, truncation=truncation,
                )
            except Exception:
                return None  # sentinel: this sample failed
            for i, tensor in enumerate(MPS_target):
                tensor.reindex_({f'input{i}': f'k{i}'})
            if use_compressed:
                M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type=noise_type)
                for i, tensor in enumerate(M_out):
                    tensor.reindex_({f'input{i}': f'k{i}'})
                loss = tensor_network_distance(
                    M_out.astype("complex128"),
                    MPS_target.astype("complex128")
                ).item()
            else:
                loss = tensor_network_distance(
                    final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"),
                    MPS_target.astype("complex128")
                ).item()
        return loss

    # ── Fast sequential path ─────────────────────────────────────────
    # We use a sequential loop here. ThreadPoolExecutor is roughly 2x slower
    # due to Python's Global Interpreter Lock (GIL). 
    # ProcessPoolExecutor cannot be used because quimb tensor networks contain
    # lambda closures that are unpicklable.
    # PyTorch's native C++ multithreading (via OMP) implicitly parallelizes the 
    # tensor operations underneath.
    raw_results = [_test_single_sample_tdme(j) for j in range(num_samples)]

    # Filter out failed samples (None sentinels)
    testing_loss_list = [r for r in raw_results if r is not None]
    n_skipped = num_samples - len(testing_loss_list)
    if n_skipped > 0:
        print(f"Skipped {n_skipped} failed testing samples.")
    testing_loss = sum(testing_loss_list) / len(testing_loss_list) if testing_loss_list else 0.0
    print("Testing loss:", testing_loss)
    return model, learning_loss, testing_loss, testing_loss_list


def Testing_TDME_Trotterization(
    N, model_layer, model_to_learn_layer, mu, gamma, J=1, t = 1, normalized = False, max_bd=64, max_err=1E-8, truncation = False, noise_type = "all", use_scheduler = True
):
    testing_loss = 0.0
    model_with_small_layer = TDME(N, model_layer, mu=mu, gamma=gamma, J=J, max_bd=max_bd, max_err=max_err)
    QMLM_MPS_output = TDME(N, model_to_learn_layer, mu=mu, gamma=gamma, J=J, max_bd=max_bd, max_err=max_err)
    num_samples = 300
    for j in range(num_samples):
        Random_MPS, weight = random_pauli_MPS(N)
        try:
            MPS_target = QMLM_MPS_output.forward(Random_MPS.copy(), t = t, truncation=truncation)
            MPS_model_with_small_layer = model_with_small_layer.forward(Random_MPS.copy(), t = t, truncation=truncation)
        except:
            print(f"Failed to generate target MPS for sample {j}, it is skipped in testing loss calculation.")
            num_samples -= 1
            continue
        loss = tensor_network_distance(
            MPS_model_with_small_layer.astype("complex128"),
            MPS_target.astype("complex128")
        ).item()
        print(f"Testing sample {j+1}/{num_samples}, loss: {loss}")
        testing_loss += loss

        """loss = tensor_network_distance(
            final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"),
            mpo_target.astype("complex128") | Random_MPS.astype("complex128")
        ).item()
        testing_loss += loss"""

    testing_loss /= num_samples
    print("Testing loss:", testing_loss)
    return testing_loss
