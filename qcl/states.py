"""Input-state builders: identity/product MPS and Pauli-string MPS bases."""
import numpy as np
import torch
import quimb.tensor as qtn



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



def Pauli_MPS_random(n, n_samples, seed=None):
    """Generate `n_samples` random Pauli MPS on n sites.

    Each MPS is drawn via `random_pauli_MPS(n)`, which picks weight w uniformly
    in [1, n] and assigns X/Y/Z uniformly at w random sites.

    If `seed` is given, NumPy and PyTorch random states are seeded once at the
    start of the loop so the returned set is reproducible.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    return [random_pauli_MPS(n)[0] for _ in range(n_samples)]



def Pauli_MPS_combined_1_and_2_full(n):
    """Concatenation of weight-1 and full weight-2 Pauli MPS bases.

    Count: 3 * n + 9 * n*(n-1)/2 (e.g., 24 + 252 = 276 at N=8).
    The intended use is to train a student on the combined basis to see
    whether having BOTH weight-1 and full weight-2 inputs in the training
    set gives better generalization than either alone.
    """
    return Pauli_MPS_weight_1(n) + Pauli_MPS_weight_2_full(n)



def Pauli_MPS_weight_2_full(n):
    """Generate the FULL weight-2 Pauli basis at n sites.

    For each ordered pair (i, j) with i < j and each combination of
    operators (op1, op2) in {X, Y, Z}^2, produce one MPS with op1 on site i,
    op2 on site j (identity elsewhere). Total count: 9 * n*(n-1)/2.

    This is a strict superset of `Pauli_MPS_weight_2`, which restricts to
    op1 = op2.

    Parameters
    ----------
    n : int
        Number of sites in the MPS.

    Returns
    -------
    list[qtn.MatrixProductState]
        Weight-2 Pauli MPS in lexicographic order over (i, j, op1, op2).
    """
    pauli_ops = ['X', 'Y', 'Z']
    MPS_list = []
    for i in range(n):
        for j in range(i + 1, n):
            for op1 in pauli_ops:
                for op2 in pauli_ops:
                    M = Identity_init(n, bond_dim=1, phys_dim=4)
                    M = operator_assignment(M, [i, j], [op1, op2])
                    MPS_list.append(M)
    return MPS_list



def Pauli_MPS_weight_2(n):
    """Generate weight-2 Pauli MPS with the SAME operator on both sites.

    For each pair (i, j) with i < j and each op in {X, Y, Z}, produce one MPS
    with `op` placed on sites i and j (identity elsewhere). Total count:
    3 * n*(n-1)/2, matching the "3N(N-1)/2" convention used as the weight-2
    analogue of the 3N weight-1 training set.

    Parameters
    ----------
    n : int
        Number of sites in the MPS.

    Returns
    -------
    list[qtn.MatrixProductState]
        Weight-2 Pauli MPS in lexicographic order over (i, j, op).
    """
    pauli_ops = ['X', 'Y', 'Z']
    MPS_list = []
    for i in range(n):
        for j in range(i + 1, n):
            for op in pauli_ops:
                M = Identity_init(n, bond_dim=1, phys_dim=4)
                M = operator_assignment(M, [i, j], [op, op])
                MPS_list.append(M)
    return MPS_list

