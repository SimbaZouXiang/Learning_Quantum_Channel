import torch
import sys
import os
import joblib

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import _process_single_mps_tdme, Pauli_MPS_weight_1, construct_TDME_unitary, construct_jump_matrices

def main():
    print("Testing Joblib...")
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
    
    args = [(i, MPS_weight1[i], T, r, all_unitary, all_jumping, 64, 1e-10, False) for i in range(4)]
    
    results = joblib.Parallel(n_jobs=2)(joblib.delayed(_process_single_mps_tdme)(a) for a in args)
    print("Joblib Success:", len(results))

if __name__ == "__main__":
    main()
