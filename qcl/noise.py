"""Single-site noise channels as Pauli transfer matrices."""
import torch

from .pauli import (_PAULI_BASIS, _EYE4, _DEPHASING_TM_X, _DEPHASING_TM_Y,
                    _DEPHASING_TM_Z, unitary_to_transfer_matrix_single_site)


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

