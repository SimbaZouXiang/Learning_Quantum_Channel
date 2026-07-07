import site
import sys
import os
import numpy as np
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import TDME_Trott as tdme

def to_numpy(x):
    """Convert torch tensor or list to numpy array without deprecation warnings."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.array(x)

N = 30
T = 3
for depolarizing_name in [2]:
    target_layer = [3, 4, 5]
    for L in target_layer:
        depolarizing_strength = depolarizing_name * 0.01
        print(f"depolarizing strength: {depolarizing_strength}, target layer: {L}")
        model, learning_loss, target_param, p_depolar, testing_loss, testing_loss_list, model_param, model_p_depolar, model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ = tdme.Learning_MPO_scheduler(N, T, L, depolarizing_strength = depolarizing_strength, epochs=200, lr=0.05, normalized = False , truncation = True,  noise_type = "depolarizing", use_compressed=True, max_bd=64)
        np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_learning_loss.npy", to_numpy(learning_loss))
        np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_testing_loss.npy", to_numpy(testing_loss))
        np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_testing_loss_list.npy", to_numpy(testing_loss_list))
        np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_target_param.npy", to_numpy(target_param))
        np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_model_param.npy", to_numpy(model_param))
        np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_model_p_depolar.npy", to_numpy(model_p_depolar))
        '''np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_model_p_dephaseX.npy", to_numpy(model_p_dephaseX))
        np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_model_p_dephaseY.npy", to_numpy(model_p_dephaseY))
        np.save(f"Depolarizing_N{N}_T{T}_L{L}_p_{depolarizing_name:03d}_model_p_dephaseZ.npy", to_numpy(model_p_dephaseZ))'''