import time
import torch
import quimb.tensor as qtn
import cotengra as ctg

# We can reuse the patch we made in TDME_Trott.py!
import sys
sys.path.append("/scratch/simba/")
from Learning_Lindbladian import TDME_Trott

print("1. Verifying that the Cotengra patch produces mathematically identical results...")
N = 10
chi = 16
mps1 = qtn.MPS_rand_state(N, bond_dim=chi)
mps2 = qtn.MPS_rand_state(N, bond_dim=chi)

d_auto = qtn.tensor_network_distance(mps1, mps2, optimize='auto')
d_ctg = TDME_Trott.tensor_network_distance(mps1, mps2)

diff = abs(d_auto - d_ctg)
print(f"   Absolute Output Difference: {diff}")
if diff < 1e-8:
    print("   [SUCCESS] The cotengra optimizer evaluates the exact same distance.")
else:
    print("   [FAILED] The distances do not match.")

print("\n2. Measuring Training loop Speedups (N=8)...")
print("   Please refer to the log tests previously run for N=8 and N=12.")
