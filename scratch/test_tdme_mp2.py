import torch
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import _process_single_mps_tdme, Pauli_MPS_weight_1, construct_TDME_unitary, construct_jump_matrices

if __name__ == "__main__":
    N = 30
    T = 3
    mu = 1.0
    gamma = [0.1]*N
    J = 1.0
    t = 1.0
    r = t/T
    print("Generating structures...")
    MPS_weight1 = Pauli_MPS_weight_1(N)
    all_unitary = construct_TDME_unitary(N, T, r=r, mu=mu, J=J)
    all_jumping = construct_jump_matrices(N, gamma, r=r)
    
    print("Running one single sample inline...")
    idx, res, err, skip = _process_single_mps_tdme((0, MPS_weight1[0], T, r, all_unitary, all_jumping, 64, 1e-10, False))
    print(f"Result returned inline? {res is not None}, Size of result approx: {sys.getsizeof(res)} bytes")
    
    import pickle
    try:
        p = pickle.dumps(res)
        print(f"Pickled size: {len(p)} bytes")
    except Exception as e:
        print(f"Pickle failed: {e}")
