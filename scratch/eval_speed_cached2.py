import time
import quimb.tensor as qtn
import cotengra as ctg

opt = ctg.ReusableHyperOptimizer(methods=['greedy'], max_repeats=16, max_time=0.5, parallel=False, progbar=False)

N = 24
chi = 64

for _ in range(5):
    mps3 = qtn.MPS_rand_state(N, bond_dim=chi)
    mps4 = qtn.MPS_rand_state(N, bond_dim=chi)
    qtn.tensor_network_distance(mps3, mps4, optimize=opt)

print(f"Num cached paths: {len(list(opt._cache.items()))}")

