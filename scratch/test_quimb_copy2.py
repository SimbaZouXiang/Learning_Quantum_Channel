import quimb.tensor as qtn
import torch

mps = qtn.MPS_rand_state(4, bond_dim=2)
mps.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
mps2 = mps.copy()

data = mps2[0].data
new_data = torch.ones_like(data)
mps2[0].modify(data=new_data)

print(type(mps[0].data))
print("MPS1 data element:", mps[0].data[0,0])
print("MPS2 data element:", mps2[0].data[0,0])
