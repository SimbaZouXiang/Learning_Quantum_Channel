import time
import quimb.tensor as qtn
import cotengra as ctg

N = 8
chi = 256
test_iters = 20

print(f"Benchmarking tensor_network_distance for N={N}, chi={chi}")
print("Generating random MPS...")
mps1 = qtn.MPS_rand_state(N, bond_dim=chi)
mps2 = qtn.MPS_rand_state(N, bond_dim=chi)

print("\n1. Testing standard 'auto' optimizer (No caching)...")
start = time.time()
for _ in range(test_iters):
    d_auto = qtn.tensor_network_distance(mps1, mps2, optimize='auto')
t_auto = (time.time() - start) / test_iters
print(f"   Average time: {t_auto:.4f} seconds/iter")

print("\n2. Testing Cotengra Reusable hyper-optimizer (Pre-computed path)...")
opt = ctg.ReusableHyperOptimizer(
    methods=['greedy', 'kahypar'],
    max_repeats=32,
    progbar=False
)

# Warmup to compute the path
print("   (Warming up / Finding path...)")
_ = qtn.tensor_network_distance(mps1, mps2, optimize=opt)

start = time.time()
for _ in range(test_iters):
    d_ctg = qtn.tensor_network_distance(mps1, mps2, optimize=opt)
t_ctg = (time.time() - start) / test_iters

print(f"   Average time: {t_ctg:.4f} seconds/iter")

speedup = t_auto / t_ctg
print(f"\nResult: Cotengra with pre-computed caching is {speedup:.2f}x faster.")
