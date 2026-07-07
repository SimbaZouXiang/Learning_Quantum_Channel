"""Trotterised time-dependent master equation (fermionic chain teacher)."""
import numpy as np
import torch
import torch.nn as nn

from .evolve import one_site_layer, two_site_layer




##########################################################################
# TDME-specific functions
#########################################################################


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
    XXpYY = np.array([[0, 0, 0, 0],
                      [0, 0, 2, 0],
                      [0, 2, 0, 0],
                      [0, 0, 0, 0]], dtype=complex)   # a† ⊗ a + a ⊗ a† after the Jordan-Wigner transformation
    num  = np.array([[0, 0], [0, 1]], dtype=complex)        # number:       |1⟩⟨1|
    if N % 2 != 0:
        raise ValueError(f"N must be even; got N={N}.")
    if not (0 <= l <= N - 1):
        raise ValueError(f"l must satisfy 0 ≤ l ≤ N-1={N-1}; got l={l}.")

    # ── Hopping: -J (a†_l ⊗ a_{l+1} + h.c.) ────────────────────────────────
    H = -J * XXpYY

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
            evolu = expm(1j * H * r/2)
            if not isinstance(evolu, torch.Tensor):
                evolu = torch.from_numpy(evolu)
            all_unitary_T.append(evolu)
        for i in range(1, N, 2):
            H = TDME_two_site_hamiltonian(i, N, mu=mu, J=J, t=r*layer)
            evolu = expm(1j * H * r)
            if not isinstance(evolu, torch.Tensor):
                evolu = torch.from_numpy(evolu)
            all_unitary_T.append(evolu)
        all_unitary.append(all_unitary_T)
    return all_unitary


def construct_jump_matrices(N: int, gamma, r: float) -> list:
    jump_matrices = []
    for i in range(N):
        beta = np.exp(-gamma[i] * r / 4)
        jump_matrix = np.diag([1, beta, beta, 1])
        if not isinstance(jump_matrix, torch.Tensor):
            jump_matrix = torch.from_numpy(jump_matrix)
        jump_matrices.append(jump_matrix)
    return jump_matrices


def Pauli_MPS_after_TDME_output_only(M, T, r, all_unitary, all_jumping,
                                     max_bd=1024, max_err=1E-10,
                                     truncation=False,
                                     track_truncation_error=False):
    """Apply the TDME Trotter circuit to an MPS.

    r is t/T, the total evolution time divided into T layers, and we want the
    output at time t = r*T.

    Parameters
    ----------
    track_truncation_error : bool, default False
        If True, after each per-layer `compress_all_simple_` call, also compute
        the MPS-overlap fidelity between the pre- and post-compression state
        and accumulate it into the returned `truncation_error` scalar. That
        requires THREE full MPS contractions per layer (one overlap + two
        norms) — for deep circuits / large N this dominates runtime, so the
        default is to skip it. Callers that discard the returned
        `truncation_error` (e.g. `_process_single_sample_test_tdme`) should
        leave this False.
    """
    L = M.L
    truncation_error = 0

    for i in range(T):
        norm_i, M = one_site_layer(M, all_jumping, truncation = truncation, max_bd = max_bd, max_err = max_err)
        norm_i, M = two_site_layer(M, all_unitary[i][:L//2], "o", truncation = truncation, max_bd = max_bd, max_err = max_err)
        M.right_canonicalize()
        norm_i, M = two_site_layer(M, all_unitary[i][L//2:], "e", truncation = truncation, max_bd = max_bd, max_err = max_err)
        norm_i, M = two_site_layer(M, all_unitary[i][:L//2], "o", truncation = truncation, max_bd = max_bd, max_err = max_err)
        M.right_canonicalize()
        norm_i, M = one_site_layer(M, all_jumping, truncation = truncation, max_bd = max_bd, max_err = max_err)
        M.right_canonicalize()

        if truncation:
            # `compress_all_simple_` runs an iterative gauge_all_simple sweep
            # before each SVD bond truncation, which dominates this loop's
            # wall time at N=30, T=30 (43+ s per layer by mid-circuit).
            # We've already done a `right_canonicalize()` immediately above,
            # so a single canonical-sweep compress is both correct and
            # ~17x faster than gauge-iterate-then-SVD.
            try:
                if track_truncation_error:
                    M_ori = M.copy()
                M.compress(form='right', max_bond=max_bd, cutoff=max_err)
                if track_truncation_error:
                    overlap = abs(M_ori.H @ M)
                    norm_orig = M_ori.norm()
                    norm_comp = M.norm()
                    fidelity = (overlap ** 2) / ((norm_orig ** 2) * (norm_comp ** 2))
                    truncation_error += 1.0 - fidelity
            except Exception as e:
                import traceback; traceback.print_exc()
                print("Compression failed at layer", i, "with max_bd =", max_bd, "and max_err =", max_err)
    return M, truncation_error



# ──────────────────────────────────────────────────────────────────────
# First-order Trotter
# Per layer:                J · U_odd · U_even          (no Strang symmetrization)
# Compare to second-order:  J^{1/2} · U_odd^{1/2} · U_even · U_odd^{1/2} · J^{1/2}
# Layer "shape" matches a typical PQC: one noise sublayer + an odd/even
# brickwall of unitaries, with no doubled gates. Useful for an apples-to-
# apples Trotter-vs-PQC comparison at fixed circuit depth.
# ──────────────────────────────────────────────────────────────────────

def construct_TDME_unitary_first_order(N: int, T: int, r: float, mu: float, J: float):
    """First-order: every two-site unitary uses the full step expm(i H r)."""
    from scipy.linalg import expm
    all_unitary = []
    for layer in range(T):
        all_unitary_T = []
        for i in range(0, N, 2):
            H = TDME_two_site_hamiltonian(i, N, mu=mu, J=J, t=r * layer)
            evolu = expm(1j * H * r)
            if not isinstance(evolu, torch.Tensor):
                evolu = torch.from_numpy(evolu)
            all_unitary_T.append(evolu)
        for i in range(1, N, 2):
            H = TDME_two_site_hamiltonian(i, N, mu=mu, J=J, t=r * layer)
            evolu = expm(1j * H * r)
            if not isinstance(evolu, torch.Tensor):
                evolu = torch.from_numpy(evolu)
            all_unitary_T.append(evolu)
        all_unitary.append(all_unitary_T)
    return all_unitary



def construct_jump_matrices_first_order(N: int, gamma, r: float) -> list:
    """First-order: full-step decay applied once per Trotter step.
    Second-order applies J^{1/2} twice per step (beta = exp(-γr/4) each, total
    decay exp(-γr/2)). First-order matches the same total dephasing per step
    by using beta = exp(-γr/2) once."""
    jump_matrices = []
    for i in range(N):
        beta = np.exp(-gamma[i] * r / 2)
        jump_matrix = np.diag([1, beta, beta, 1])
        if not isinstance(jump_matrix, torch.Tensor):
            jump_matrix = torch.from_numpy(jump_matrix)
        jump_matrices.append(jump_matrix)
    return jump_matrices



def Pauli_MPS_after_TDME_output_only_first_order(M, T, r, all_unitary, all_jumping,
                                                 max_bd=1024, max_err=1E-10,
                                                 truncation=False):
    """First-order Trotter forward pass: per layer J → U_odd → U_even, once."""
    L = M.L
    truncation_error = 0

    for i in range(T):
        norm_i, M = one_site_layer(M, all_jumping, truncation=truncation, max_bd=max_bd, max_err=max_err)
        norm_i, M = two_site_layer(M, all_unitary[i][:L // 2], "o", truncation=truncation, max_bd=max_bd, max_err=max_err)
        M.right_canonicalize()
        norm_i, M = two_site_layer(M, all_unitary[i][L // 2:], "e", truncation=truncation, max_bd=max_bd, max_err=max_err)
        M.right_canonicalize()

        if truncation:
            try:
                M.compress(form='right', max_bond=max_bd, cutoff=max_err)
            except Exception as e:
                import traceback; traceback.print_exc()
                print("Compression failed at layer", i, "with max_bd =", max_bd, "and max_err =", max_err)
    return M, truncation_error



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
        M_output, truncation_error = Pauli_MPS_after_TDME_output_only(M_input, self.T, r=r, all_unitary=all_unitary, all_jumping=all_jumping, max_bd=self.max_bd, max_err=self.max_err, truncation=truncation)
        return M_output

