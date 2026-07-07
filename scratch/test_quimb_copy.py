import quimb.tensor as qtn
import torch

mps = qtn.MPS_rand_state(4, bond_dim=2)
mps2 = mps.copy()
mps2[0].data[0,0] = 999.0
print(mps[0].data[0,0])
