import time
import sys
import torch
import quimb.tensor as qtn

sys.path.append("/scratch/simba/")
from Learning_Lindbladian import TDME_Trott

if __name__ == '__main__':
    print(f"\n--- Running Training Epoch with Cotengra Optimizer (use_compressed=False) ---")
    t0 = time.time()
    TDME_Trott.Learning_TDME_scheduler(
        N=8, MPO_layer=3, model_to_learn_layer=3, mu=0.5, gamma=[0.1]*8, J=1.0, t=0.6,
        epochs=1, lr=0.01, normalized=False, max_bd=64, max_err=1E-6, 
        truncation=True, noise_type="all", use_scheduler=False,
        use_compressed=False, num_threads=4   # <<< CHANGED TO FALSE
    )
    elapsed = time.time() - t0
    print(f"Cotengra Complete: {elapsed:.2f} seconds")
