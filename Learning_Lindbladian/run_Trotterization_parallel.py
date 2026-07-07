import os
import sys

# Pin thread counts BEFORE heavy imports to prevent CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import time
import numpy as np

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import TDME_Trott as tdme

def main():
    parser = argparse.ArgumentParser(description="Run Testing_TDME_Trotterization_parallel and save results")
    parser.add_argument("--threads", type=int, default=64, help="Number of threads for parallel testing")
    parser.add_argument("--samples", type=int, default=300, help="Number of samples to test")
    args = parser.parse_args()

    # Define hyperparameters based on typical simulation runs
    N = 30
    model_layer = 3
    model_to_learn_layer = 30
    mu = 1.0  # Can be float or a callable function
    J = 1.0
    max_bd = 64
    max_err = 1E-8
    truncation = True
    noise_type = "dephasing"

    target_times = [0.6, 0.8]
    gamma_names = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]

    # Output directory
    out_dir = os.path.join(SCRIPT_DIR, "Trotterization")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Results will be stored in: {out_dir}")
    print(f"Running with {args.threads} threads for {args.samples} samples per configuration...")

    for target_time in target_times:
        for gamma_name in gamma_names:
            gamma = [gamma_name * 0.01] * N
            
            print(f"\n--- Testing target_time={target_time}, gamma={gamma[0]:.2f} ---")
            start_time = time.time()
            
            testing_loss, testing_loss_list = tdme.Testing_TDME_Trotterization_parallel(
                N=N,
                model_layer=model_layer,
                model_to_learn_layer=model_to_learn_layer,
                mu=mu,
                gamma=gamma,
                J=J,
                t=target_time,
                max_bd=max_bd,
                max_err=max_err,
                truncation=truncation,
                noise_type=noise_type,
                num_samples=args.samples,
                num_threads=args.threads
            )
            
            # Format filename explicitly
            base_filename = f"N{N}_T{model_layer}_L{model_to_learn_layer}_mu{float(mu)}_gamma{round(gamma[0], 2)}_t{target_time}"
            file_prefix = os.path.join(out_dir, base_filename)
            
            # Store values
            np.save(file_prefix + "_testing_loss.npy", np.array(testing_loss))
            np.save(file_prefix + "_testing_loss_list.npy", np.array(testing_loss_list))
            
            end_time = time.time()
            print(f"-> Saved to {file_prefix}_testing_loss.npy")
            print(f"-> Time taken: {end_time - start_time:.2f} seconds. Mean Loss: {testing_loss:.5f}")

if __name__ == "__main__":
    main()
