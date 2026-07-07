import cotengra as ctg
import quimb.tensor as qtn
from quimb.tensor.fitting import tensor_network_distance
import torch

a = qtn.MPS_rand_state(8, bond_dim=4).astype("complex128")
b = qtn.MPS_rand_state(8, bond_dim=4).astype("complex128")

opt = ctg.ReusableHyperOptimizer(
    methods=['greedy', 'kahypar'], 
    max_repeats=16, 
    max_time=1.0,
    parallel=True,
    progbar=False
)

res = tensor_network_distance(a, b, optimize=opt)
print(res)
