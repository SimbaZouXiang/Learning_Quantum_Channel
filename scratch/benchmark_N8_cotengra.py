import sys
import time
import quimb.tensor as qtn
import torch
import numpy as np

sys.path.append("/scratch/simba/")
from Learning_Lindbladian.TDME_Trott import Learning_TDME_scheduler

def run_benchmark():
    N = 8
    mu = 0.5
    gamma = [0.1]*N
    J = 1.0
    t = 0.6
    
    print(f"Benchmarking Learning_TDME_scheduler with N={N}...")
    start = time.time()
    
    Learning_TDME_scheduler(
        N=N, MPO_layer=2, model_to_learn_layer=2, mu=mu, gamma=gamma, J=J, t=t,
        epochs=1, lr=0.01, normalized=False, max_bd=16, max_err=1E-6, 
        truncation=True, noise_type="all", use_scheduler=False,
        use_compressed=True, num_threads=4  # fewer threads for isolation testing
    )
    
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")

if __name__ == '__main__':
    run_benchmark()
