import os
import sys
import json

# Set NUMBA_CACHE_DIR to /tmp to prevent "no locator available" errors in cluster environments
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

# Add the parent directory to the front of sys.path to ensure local imports succeed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import get_input_and_output_MPS_TDME
from Learning_unitary.load_training_data import save_mps

def generate_and_save_dataset(
    N, T, mu_val, gamma_val, J, t, max_bd, max_err, num_threads, out_dir
):
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating dataset for N={N}, T={T}, mu={mu_val}, gamma={gamma_val} with {num_threads} threads...")
    
    # Let mu be a constant function if we want, or simple float
    mu = lambda t_val: mu_val
    gamma = [gamma_val] * N
    
    MPS_weight1, target_mps_list, truncation_error = get_input_and_output_MPS_TDME(
        N, T, mu=mu, gamma=gamma, J=J, t=t,
        truncation=True, max_bd=max_bd, max_err=max_err,
        num_threads=num_threads
    )
    
    print(f"Saving {len(MPS_weight1)} inputs and {len(target_mps_list)} outputs to {out_dir}...")
    
    for i, mps in enumerate(MPS_weight1):
        save_mps(mps, os.path.join(out_dir, f"input_MPS_{i}.npz"))
        
    for i, mps in enumerate(target_mps_list):
        save_mps(mps, os.path.join(out_dir, f"target_MPS_{i}.npz"))
        
    # Save parameters
    params = {
        "N": N, "T": T, "mu": mu_val, "gamma": gamma_val, 
        "J": J, "t": t, "max_bd": max_bd, "max_err": max_err,
        "num_samples": len(MPS_weight1), "truncation_error": float(truncation_error)
    }
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
        
    print(f"Done! Data saved to {out_dir}")

if __name__ == "__main__":
    '''import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--t", type=float, default=0.01)
    parser.add_argument("--max_bd", type=int, default=128)
    parser.add_argument("--max_err", type=float, default=1e-8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "Learning_data", f"N{args.N}_T{args.T}_mu{args.mu}_gamma{args.gamma}")
    
    generate_and_save_dataset(
        args.N, args.T, args.mu, args.gamma, args.J, args.t, 
        args.max_bd, args.max_err, args.threads, out_dir
    )'''
    N = 30
    T = 30
    mu = 1.0
    J = 1.0
    max_bd = 64
    max_err = 1e-8
    num_threads = 8
    gamma_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
    t_values  = [1.0, 2.0]
    for gamma in gamma_values:
        for t in t_values:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.join(base_dir, "Learning_data", f"N{N}_T{T}_mu{mu}_gamma{gamma}_t{t}")
            generate_and_save_dataset(
                N, T, mu, gamma, J, t, 
                max_bd, max_err, num_threads, out_dir
            )
