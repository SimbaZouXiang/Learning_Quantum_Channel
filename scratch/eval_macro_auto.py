import time
import sys
import torch
import quimb.tensor as qtn

sys.path.append("/scratch/simba/")
from Learning_Lindbladian import TDME_Trott
# Unpatch dynamically to revert to default 'opt-einsum' (auto)
from quimb.tensor.fitting import tensor_network_distance as orig_tnd
TDME_Trott.tensor_network_distance = orig_tnd

if __name__ == '__main__':
    print(f"\n--- Running Training Epoch with Original Opt-Einsum (use_compressed=False) ---")
    t0 = time.time()
    TDME_Trott.Learning_TDME_scheduler(
        N=8, MPO_layer=3, model_to_learn_layer=3, mu=0.5, gamma=[0.1]*8, J=1.0, t=0.6,
        epochs=1, lr=0.01, normalized=False, max_bd=64, max_err=1E-6, 
        truncation=True, noise_type="all", use_scheduler=False,
        use_compressed=False, num_threads=4 
    )
    elapsed = time.time() - t0
    print(f"Auto Complete: {elapsed:.2f} seconds")
