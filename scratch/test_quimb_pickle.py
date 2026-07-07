import pickle
import quimb.tensor as qtn
import torch

try:
    M = qtn.MPS_rand_state(4, bond_dim=2)
    p = pickle.dumps(M)
    M2 = pickle.loads(p)
    print(f"Pickle Success! size={len(p)}")
except Exception as e:
    print(f"Pickle Failed: {e}")
