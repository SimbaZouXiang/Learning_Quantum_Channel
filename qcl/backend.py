"""Shared numerical backend: cotengra path cache, tensor_network_distance
wrapper, and the global torch.linalg.svd safety monkey-patch.

This module MUST be imported before any quimb compression runs — importing
qcl (or the TDME_Trott shim) does this automatically.
"""
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
import time
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.fitting import tensor_network_distance as _tensor_network_distance
import cotengra as ctg

# Global optimizer for tensor network distances
# ReusableHyperOptimizer caches the contraction paths and accelerates subsequent distance evaluations
opt = ctg.ReusableHyperOptimizer(
    methods=['greedy'],
    max_repeats=16,
    max_time=1.0,
    # Sequential path search: every worker in a ProcessPoolExecutor would
    # otherwise spin up its own cotengra pool, compounding oversubscription.
    # Paths are cached across calls anyway, so this only touches first-hit
    # cost — negligible vs the autograd steps.
    parallel=False,
    progbar=False,
)

def tensor_network_distance(*args, optimize=opt, **kwargs):
    return _tensor_network_distance(*args, optimize=optimize, **kwargs)

from ncon import ncon
import torch
import torch.multiprocessing as mp

# ---- Add this patch to fix SVD non-convergence errors ----
# Also adds a tiny deterministic jitter before the forward SVD when the input
# has near-degenerate singular values. SVD's backward formula contains
# 1/(s_i^2 - s_j^2) terms and blows up to NaN when two singular values are
# close. For the Pauli-basis transfer-matrix / bond-compression tensors used
# in Pauli_MPS_after_QMLM_output_only, exact degeneracies are common, so we
# perturb the input by a tiny multiple of its Frobenius norm to break them
# without meaningfully changing the compressed result.
_orig_svd = torch.linalg.svd
_SVD_JITTER_REL = 1e-6  # relative to max |A|

def _safe_svd(A, **kwargs):
    if A.requires_grad and A.ndim >= 2 and A.shape[-1] > 1 and A.shape[-2] > 1 and A.numel() > 0:
        # Break exact-repeat singular values before SVD. Backward uses
        # 1/(s_i^2 - s_j^2), which yields NaN when two singular values are
        # equal — common for Pauli-basis transfer matrices with structured
        # symmetries. NB: this alone does NOT make quimb's compressed MPS
        # training fully stable; deep circuits still produce NaN grads
        # through cascaded SVDs. The jitter helps in shallow cases only.
        scale = A.detach().abs().amax()
        if scale.item() > 0:
            noise = torch.randn_like(A.detach()) * (_SVD_JITTER_REL * scale)
            A = A + noise
    def _scipy_gesvd_fallback():
        # gesvd is slower but far more robust than the default gesdd driver.
        device = A.device
        A_np = A.detach().cpu().numpy()
        if not np.all(np.isfinite(A_np)):
            # Nothing can rescue a non-finite input; fail with a message that
            # names the real problem instead of scipy's opaque ValueError.
            raise RuntimeError(
                "SVD input contains non-finite values (NaN/inf). This usually "
                "means NaN gradients from an earlier step poisoned the model "
                "parameters — see the finite-gradient guard in qcl.training."
            )
        full_matrices = kwargs.get('full_matrices', True)
        import scipy.linalg
        try:
            U, S, Vh = scipy.linalg.svd(A_np, full_matrices=full_matrices, lapack_driver='gesvd')
        except Exception:
            A_np = A_np + 1e-12 * np.random.randn(*A_np.shape) + 1j * 1e-12 * np.random.randn(*A_np.shape) if np.iscomplexobj(A_np) else A_np + 1e-12 * np.random.randn(*A_np.shape)
            U, S, Vh = scipy.linalg.svd(A_np, full_matrices=full_matrices, lapack_driver='gesvd')
        return (torch.from_numpy(U).to(device),
                torch.from_numpy(S).to(device),
                torch.from_numpy(Vh).to(device))

    try:
        out = _orig_svd(A, **kwargs)
    except RuntimeError as e:
        err_msg = str(e).lower()
        if 'converge' in err_msg or 'gesvd' in err_msg or 'error code: 3' in err_msg or 'ill-conditioned' in err_msg:
            return _scipy_gesvd_fallback()
        raise e
    # LAPACK's gesdd (torch's default driver) can return NaN/inf SILENTLY on
    # ill-conditioned inputs instead of raising. Detect that and redo the
    # decomposition with the robust gesvd driver. Only inspect S (cheap, and
    # gesdd failures always show up there); skip when autograd is tracing —
    # the fallback would detach the graph, and training inputs are jittered
    # above anyway.
    if not A.requires_grad and not torch.isfinite(out[1]).all():
        return _scipy_gesvd_fallback()
    return out
torch.linalg.svd = _safe_svd
# ------------------------------------------------------------

import warnings
import torch.optim as optim
#from quimb.tensor import qr
import cotengra as ctg
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
