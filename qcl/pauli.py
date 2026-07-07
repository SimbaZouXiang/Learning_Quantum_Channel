"""Pauli basis constants, unitary→transfer-matrix conversions, SU(4) construction.

All quantum channels in this package are represented in the vectorised Pauli
basis (physical dimension 4: I, X, Y, Z). The constants below are built once
at import time — do not recompute them per call.
"""
import numpy as np
import torch


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


# Partial-trace vector for a Pauli-basis (phys-dim 4) leg: picks the Identity
# component. Matches Identity_init which uses [1,0,0,0] for the Identity operator.
# Stored as real float64 so that contracting with a float tensor (e.g. a raw
# bath site before forward has cast to complex) does not trigger a spurious
# complex-to-real cast warning; promote to complex via .to(...) where needed.
_TRACE_VEC_PAULI = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)



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
    # Batched QR: single kernel launch for T*L matrices instead of a Python double loop.
    # The j=L-1 slot is never consumed by downstream two-site layer loops (they iterate
    # range(site0, N-1, 2)), so filling it with a valid SU(4) here is behaviourally safe.
    params_sq = param_list.reshape(T * L, 4, 4)
    Q, R = torch.linalg.qr(params_sq)
    diag_R = R.diagonal(dim1=-2, dim2=-1)
    phases = torch.where(
        diag_R.abs() > 0,
        diag_R / diag_R.abs(),
        torch.ones_like(diag_R),
    )
    Q = Q * phases.conj().unsqueeze(-2)
    return Q.reshape(T, L, 4, 4)



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


def unitary_to_transfer_matrix_two_site_truncated_batched(Us):
    """Batched version: Us of shape (..., 4, 4) → real TMs of shape (..., 16, 16).

    Called once per epoch from apply_mpo_to_mps_compressed to avoid redundantly
    recomputing the same transfer matrix inside evol_two_site for every sample.
    """
    Ps2 = _PAULI_BASIS_2SITE
    Uc = Us.conj().transpose(-1, -2)
    T = torch.einsum('...ab, bci, ...cd, daj -> ...ij', Us, Ps2, Uc, Ps2) / 4
    return torch.real(T)

