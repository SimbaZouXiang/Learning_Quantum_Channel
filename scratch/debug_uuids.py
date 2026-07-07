import sys
import torch
import quimb.tensor as qtn
sys.path.append("/scratch/simba/")
from Learning_Lindbladian import TDME_Trott

# Unpatch dynamically to revert to default 'opt-einsum' (auto)
from quimb.tensor.fitting import tensor_network_distance as orig_tnd
TDME_Trott.tensor_network_distance = orig_tnd

N = 4
mu = 0.5
gamma = [0.1]*4

model = TDME_Trott.QMLM(N, 2)
MPS_weight1 = TDME_Trott.Pauli_MPS_weight_1(N)

mpo1 = model.get_MPO(noise_type="all")
mpo2 = model.get_MPO(noise_type="all")

for t in mpo1.tensors:
    if t.inds not in [t2.inds for t2 in mpo2.tensors]:
        print("Different indices found in MPO tensors!", t.inds)

M_in1 = mpo1 | MPS_weight1[0]
M_in2 = mpo2 | MPS_weight1[0]

print("Total indices match?", M_in1.outer_inds() == M_in2.outer_inds(), M_in1.inner_inds() == M_in2.inner_inds())
print("M_in1 inner:", list(M_in1.inner_inds())[:5])
print("M_in2 inner:", list(M_in2.inner_inds())[:5])

