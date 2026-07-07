"""
Utility functions for saving and loading quimb MatrixProductState objects
to/from .npz files.

Usage:
    from load_training_data import save_mps, load_mps

    save_mps(mps, "path/to/file.npz")
    mps_rebuilt = load_mps("path/to/file.npz")
"""

import numpy as np
import quimb.tensor as qtn
import torch


def save_mps(mps, filepath):
    """Save a quimb MatrixProductState to a .npz file.

    Each site tensor's numpy/torch array is stored as 'site_0', 'site_1', ...
    The file can be loaded back with ``load_mps``.

    Parameters
    ----------
    mps : qtn.MatrixProductState
        The MPS to save.
    filepath : str
        Output path (should end in .npz).
    """
    arrays = {}
    for i in range(mps.L):
        data = mps[i].data
        # Convert torch tensors to numpy
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        arrays[f"site_{i}"] = np.array(data)
    np.savez_compressed(filepath, **arrays)


def load_mps(filepath, input = True):
    """Load a quimb MatrixProductState from a .npz file saved by ``save_mps``.

    The MPS is reconstructed with sequential physical indices 'k0', 'k1', ...
    matching the convention used by ``Pauli_MPS_weight_1`` outputs after
    reindexing.

    Parameters
    ----------
    filepath : str
        Path to the .npz file.

    Returns
    -------
    qtn.MatrixProductState
        The reconstructed MPS.
    """
    data = np.load(filepath, allow_pickle=False)
    num_sites = len(data.files)
    arrays = [torch.from_numpy(data[f"site_{i}"]) for i in range(num_sites)]
    M = qtn.MatrixProductState(arrays)
    for i, tensor in enumerate(M):
        indices = tensor.inds
        M[i].reindex({indices[1]: indices[2], indices[2]: indices[1]}, inplace=True)
        if input:
            M[i].reindex_({f'k{i}': f'input{i}'})
    if input:
        M[i].reindex({indices[2]: 'void2'}, inplace=True)

    return M
