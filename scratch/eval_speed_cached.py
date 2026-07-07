import time
import torch
import quimb.tensor as qtn
import cotengra as ctg

# 1. Setup Optimizer (Only Greedy, No KaHyPar since kahypar module is missing from user's env)
opt = ctg.ReusableHyperOptimizer(
    methods=['greedy'],
    max_repeats=16, 
    max_time=0.5,
    parallel=False,
    progbar=False
)

N = 24
chi = 64
mps1 = qtn.MPS_rand_state(N, bond_dim=chi, tags='mps1')
mps2 = qtn.MPS_rand_state(N, bond_dim=chi, tags='mps2')

# Do it purely for 1D MPS, but re-init random states
# Does `qtn.tensor_network_distance` reuse same eq?
t0 = time.time()
for _ in range(5):
    # Generates SAME indices structure under the hood?
    qtn.tensor_network_distance(mps1, mps2, optimize=opt)
t_ctg1 = (time.time() - t0) / 5

t0 = time.time()
for _ in range(5):
    qtn.tensor_network_distance(mps1, mps2, optimize=opt)
t_ctg2 = (time.time() - t0) / 5

print(f"Cotengra First 5: {t_ctg1:.4f} s/iter")
print(f"Cotengra Next 5:  {t_ctg2:.4f} s/iter")

print(f"\nOptimizer Cache size: {len(opt._cache)}")

