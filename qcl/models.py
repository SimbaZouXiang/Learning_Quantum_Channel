"""Trainable student (QMLM) and non-trainable teacher (QMLM_output_only) models."""
import torch
import torch.nn as nn

from .backend import tensor_network_distance
from .pauli import (construct_all_SU4,
                    unitary_to_transfer_matrix_two_site_truncated_batched)
from .mpo import Build_QMLM_MPO, Pauli_MPS_after_QMLM
from .evolve import Pauli_MPS_after_QMLM_output_only

import quimb.tensor as qtn


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

    def get_MPO_without_depolarizing_noise(self):
        """Dephasing-only MPO: unitary layers + dephasing X/Y/Z channels, no
        depolarizing layer. Called by the Learning_MPO_dephasing_noise_only*
        training loops on their non-compressed path (was missing — those loops
        crashed with AttributeError whenever use_compressed=False)."""
        return self.get_MPO(noise_type="dephasing")

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
        all_tms = unitary_to_transfer_matrix_two_site_truncated_batched(all_U)
        M_output = Pauli_MPS_after_QMLM_output_only(
            M_input.copy(), self.layers, all_U,
            self.p_depolar, self.p_dephaseX, self.p_dephaseY, self.p_dephaseZ,
            max_bd=max_bond, max_err=cutoff, truncation=True, noise_type=noise_type,
            all_tms=all_tms,
        )
        return M_output

    def forward_compressed_depolarizing_only(self, M_input, max_bond=64, cutoff=1e-10):
        """Same as forward_compressed but with depolarizing noise only."""
        all_U = construct_all_SU4(self.N, self.layers, self.params)
        all_tms = unitary_to_transfer_matrix_two_site_truncated_batched(all_U)
        M_output = Pauli_MPS_after_QMLM_output_only(
            M_input.copy(), self.layers, all_U,
            self.p_depolar,
            max_bd=max_bond, max_err=cutoff, truncation=True, noise_type="depolarizing",
            all_tms=all_tms,
        )
        return M_output



def apply_mpo_to_mps_compressed(model, input_mps_list, max_bond=64, cutoff=1e-10, noise_type="all", truncation=False):
    """Apply the QMLM model to a list of input MPS, returning compressed output MPS.

    This is more efficient than the pattern ``model.get_MPO() | mps`` because:
    1. The circuit is applied layer-by-layer with qr truncation, keeping
       the bond dimension at most *max_bond*.
    2. The returned objects are proper MPS — ``tensor_network_distance``
       between two MPS is O(N D^3), vs exponential for an uncontracted TN.

    The SU(4) unitaries are constructed once and re-used for every sample.

    Parameters:
        model (QMLM): Trained QMLM model.
        input_mps_list (list): List of input MPS.
        max_bond (int): Maximum bond dimension after compression.
        cutoff (float): qr truncation cutoff.
        noise_type (str): "all", "depolarizing", "dephasing", or "none".

    Returns:
        list: List of compressed output MPS (with physical indices 'k0','k1',…).
    """
    all_U = construct_all_SU4(model.N, model.layers, model.params)
    # Precompute 2-site transfer matrices ONCE per epoch instead of per-sample.
    # Autograd flows through this batched op, and the per-sample evol_two_site
    # calls read shared tensors from it.
    all_tms = unitary_to_transfer_matrix_two_site_truncated_batched(all_U)
    output_list = []
    for mps in input_mps_list:
        M_out = Pauli_MPS_after_QMLM_output_only(
            mps.copy(), model.layers, all_U,
            model.p_depolar, model.p_dephaseX, model.p_dephaseY, model.p_dephaseZ,
            max_bd=max_bond, max_err=cutoff, truncation=truncation, noise_type=noise_type,
            all_tms=all_tms,
        )
        # Reindex from 'input{i}' to 'k{i}' to match targets
        for i, tensor in enumerate(M_out):
            tensor.reindex_({f'input{i}': f'k{i}'})
        output_list.append(M_out)
    return output_list


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

