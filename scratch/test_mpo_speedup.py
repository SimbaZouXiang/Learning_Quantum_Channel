import time
import quimb.tensor as qtn
import cotengra as ctg

N = 8
chi = 64
chi_mpo = 16
test_iters = 20

print(f"Benchmarking <MPS | MPO | MPS> for N={N}, MPS chi={chi}, MPO chi={chi_mpo}")
print("Generating random Tensor Networks...")
mps1 = qtn.MPS_rand_state(N, bond_dim=chi)
mpo = qtn.MPO_rand_herm(N, bond_dim=chi_mpo)
mps2 = qtn.MPS_rand_state(N, bond_dim=chi)

# Create the full tensor network for expectation value
# Using the standard quimb align notation
tn = (mps1.H | mpo | mps2)

print("\n1. Testing standard 'auto' optimizer (No caching)...")
start = time.time()
for _ in range(test_iters):
    val_auto = tn.contract(optimize='auto')
t_auto = (time.time() - start) / test_iters
print(f"   Average time: {t_auto:.4f} seconds/iter")

print("\n2. Testing Cotengra Reusable hyper-optimizer (Pre-computed path)...")
# Note that cotengra will cache the result if structural ids match, but here we just
# create a new optimizer cache
opt = ctg.ReusableHyperOptimizer(
    methods=['kahypar', 'greedy'],
    max_repeats=32,
    progbar=False
)

print("   (Warming up to compute contraction path...)")
_ = tn.contract(optimize=opt)

start = time.time()
for _ in range(test_iters):
    val_ctg = tn.contract(optimize=opt)
t_ctg = (time.time() - start) / test_iters

print(f"   Average time: {t_ctg:.4f} seconds/iter")

speedup = t_auto / t_ctg
print(f"\nResult: Cotengra with pre-computed caching is {speedup:.2f}x faster.")
