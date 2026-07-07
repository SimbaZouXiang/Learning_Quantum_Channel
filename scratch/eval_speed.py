import time
import torch
import quimb.tensor as qtn
import cotengra as ctg

# 1. Setup Optimizer
opt = ctg.ReusableHyperOptimizer(
    methods=['kahypar', 'greedy'], 
    max_repeats=16, 
    max_time=1.0,
    parallel=True,
    progbar=False
)

# 2. Benchmark MPS-MPS overlap (Simple 1D contraction)
N = 20
chi = 32
mps1 = qtn.MPS_rand_state(N, bond_dim=chi, tags='mps1')
mps2 = qtn.MPS_rand_state(N, bond_dim=chi, tags='mps2')

print(f"--- Benchmarking MPS vs MPS distance (N={N}, chi={chi}) ---")
# Warmup
_ = qtn.tensor_network_distance(mps1, mps2, optimize='auto')

t0 = time.time()
for _ in range(5):
    d_auto = qtn.tensor_network_distance(mps1, mps2, optimize='auto')
t_auto = (time.time() - t0) / 5
print(f"Default (opt-einsum):  {t_auto:.4f} seconds/iter")

t0 = time.time()
for _ in range(5):
    d_ctg = qtn.tensor_network_distance(mps1, mps2, optimize=opt)
t_ctg = (time.time() - t0) / 5
print(f"Cotengra (Reusable):   {t_ctg:.4f} seconds/iter")
print(f"Results match: {abs(d_auto - d_ctg) < 1e-8}\n")

# 3. Benchmark MPO|MPS overlap (2D Contraction)
# This mimics final_mpo_fit | Random_MPS
mpo1 = qtn.MPO_rand(N, bond_dim=4, tags='mpo1')
mpo2 = qtn.MPO_rand(N, bond_dim=4, tags='mpo2')
tn1 = mpo1 | mps1
tn2 = mpo2 | mps1

print(f"--- Benchmarking (MPO|MPS) vs (MPO|MPS) distance (N={N}, chi={chi}) ---")
# Warmup
try:
    _ = qtn.tensor_network_distance(tn1, tn2, optimize='auto')
except Exception as e:
    print("Default opt-einsum failed or is too slow for warmup!")

t0 = time.time()
for _ in range(3):
    d_auto = qtn.tensor_network_distance(tn1, tn2, optimize='auto')
t_auto = (time.time() - t0) / 3
print(f"Default (opt-einsum):  {t_auto:.4f} seconds/iter")

t0 = time.time()
for _ in range(3):
    d_ctg = qtn.tensor_network_distance(tn1, tn2, optimize=opt)
t_ctg = (time.time() - t0) / 3
print(f"Cotengra (Reusable):   {t_ctg:.4f} seconds/iter")
print(f"Results match: {abs(d_auto - d_ctg) < 1e-8}\n")

