import time
import quimb.tensor as qtn
import cotengra as ctg

N = 8
chi = 64

# Warm up optimizers
opt = ctg.ReusableHyperOptimizer(
    methods=['greedy'], max_repeats=16, max_time=1.0, parallel=True, progbar=False
)

mps1 = qtn.MPS_rand_state(N, bond_dim=chi)
mps2 = qtn.MPS_rand_state(N, bond_dim=chi)
mps3 = qtn.MPS_rand_state(N, bond_dim=chi)

print(f"Benchmarking tensor_network_distance for N={N}, chi={chi}...")

# Auto Warnup
_ = qtn.tensor_network_distance(mps1, mps2, optimize='auto')
t0 = time.time()
for _ in range(20):
    qtn.tensor_network_distance(mps1, mps2, optimize='auto')
t_auto = (time.time() - t0) / 20
print(f"Auto (opt-einsum):  {t_auto:.5f} sec/iter")

# Cotengra Warmup (first time builds cache)
_ = qtn.tensor_network_distance(mps1, mps2, optimize=opt)
t0 = time.time()
for _ in range(20):
    # Pass same structure, different tensors
    qtn.tensor_network_distance(mps2, mps3, optimize=opt)
t_ctg = (time.time() - t0) / 20
print(f"Cotengra (cached):  {t_ctg:.5f} sec/iter")

if t_ctg < t_auto:
    print(f"Speedup: {t_auto/t_ctg:.2f}x faster with Cotengra")
else:
    print(f"Cotengra is {t_ctg/t_auto:.2f}x slower for this size/chi")

