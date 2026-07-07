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

max_bd = 16
max_err = 1e-5
M = qtn.TensorNetwork([
    qtn.Tensor(data=np.random.rand(2, 50, 2), inds=('a', 'b', 'c')),
    qtn.Tensor(data=np.random.rand(2, 50, 2), inds=('c', 'd', 'e')),
    qtn.Tensor(data=np.random.rand(2, 50, 2), inds=('e', 'f', 'g'))])

trunc_info = {}
M = qtn.tensor_network_1d_compress(
    M,
    method="direct",        # or "dm"
    max_bond=max_bd,           # hard maximum bond dimension
    cutoff=max_err,           # error / truncation threshold
    cutoff_mode="rsum2",    # "discarded weight" style
    permute_arrays=False,
    compress_opts={'info': trunc_info}
)
error = trunc_info.get('error')
print("Error:", error)
