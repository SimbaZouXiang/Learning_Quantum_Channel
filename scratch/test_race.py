import torch
import sys
import os
import concurrent.futures

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import _process_single_mps_tdme, Pauli_MPS_weight_1, construct_TDME_unitary, construct_jump_matrices
import time

def main():
    print("Testing ThreadPool Race fix...")
    N = 30
    T = 3
    mu = 1.0
    gamma = [0.1]*N
    J = 1.0
    t = 1.0
    r = t/T
    
    MPS_weight1 = Pauli_MPS_weight_1(N)
    all_unitary = construct_TDME_unitary(N, T, r=r, mu=mu, J=J)
    all_jumping = construct_jump_matrices(N, gamma, r=r)
    
    # Pre-cast to tensors to prevent race condition
    for u_layer in all_unitary:
        for i in range(len(u_layer)):
            u_layer[i] = torch.tensor(u_layer[i], dtype=torch.complex128)
    for i in range(len(all_jumping)):
        all_jumping[i] = torch.tensor(all_jumping[i], dtype=torch.complex128)
        
    args = [(i, MPS_weight1[i], T, r, all_unitary, all_jumping, 64, 1e-10, False) for i in range(4)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_process_single_mps_tdme, args))
    print(f"ThreadPool Success: {len(results)}, time={time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
