import time
import inspect
import quimb.tensor as qtn
import cotengra as ctg
from quimb.tensor.fitting import tensor_network_distance

def run_benchmark():
    N = 8
    bond_dim = 16
    
    print(f"Generating random MPO and MPS for N={N}, max_bd={bond_dim}...")
    mpo = qtn.MPO_rand(N, bond_dim)
    mps = qtn.MPS_rand_state(N, bond_dim)

    tn1 = (mpo | mps).astype("complex128")
    tn2 = (qtn.MPO_rand(N, bond_dim) | mps).astype("complex128")

    opt = ctg.ReusableHyperOptimizer(
        progbar=False,
        reconf_opts={},
        max_repeats=16,
        max_time='eq:2',
        parallel=True
    )

    print("\n--- Benchmarking ---")
    
    _ = tensor_network_distance(tn1, tn2)
    _ = tensor_network_distance(tn1, tn2, optimize=opt)

    num_iters = 10
    
    start = time.perf_counter()
    for _ in range(num_iters):
        dist_default = tensor_network_distance(tn1, tn2)
    time_default = time.perf_counter() - start
    print(f"Default execution time ({num_iters} iters): {time_default:.4f} seconds")

    start = time.perf_counter()
    for _ in range(num_iters):
        dist_opt = tensor_network_distance(tn1, tn2, optimize=opt)
    time_opt = time.perf_counter() - start
    print(f"Cotengra execution time ({num_iters} iters): {time_opt:.4f} seconds")
    
    speedup = time_default / time_opt
    print(f"Speedup factor: {speedup:.2f}x\n")

if __name__ == "__main__":
    run_benchmark()
