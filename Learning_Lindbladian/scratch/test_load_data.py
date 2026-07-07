import os
import sys
import json
import quimb as qu
import quimb.tensor as qtn

# Add the parent directory to the front of sys.path to ensure local imports succeed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Learning_unitary.load_training_data import load_mps

def test_load_data(data_dir):
    print(f"Loading datasets from {data_dir}...")
    
    params_file = os.path.join(data_dir, "params.json")
    if not os.path.exists(params_file):
        print(f"Error: {params_file} does not exist.")
        return
        
    with open(params_file, "r") as f:
        params = json.load(f)
    print(f"Loaded params: {params}")
    
    num_samples = params.get("num_samples", 0)
    print(f"Total samples to load: {num_samples}")
    
    # We will just load the first 2 as a quick demonstration
    samples_to_check = min(2, num_samples)
    
    for i in range(samples_to_check):
        input_path = os.path.join(data_dir, f"input_MPS_{i}.npz")
        target_path = os.path.join(data_dir, f"target_MPS_{i}.npz")
        
        # Using the utility function
        in_mps = load_mps(input_path)
        out_mps = load_mps(target_path)
        
        distance = qtn.tensor_network_distance(in_mps, out_mps)
        print(f"Tensor network distance: {distance}")
        
        print(f"\n--- Sample {i} ---")
        print(f"Input MPS:  {type(in_mps)} with {in_mps.L} sites.")
        print(f"Target MPS: {type(out_mps)} with {out_mps.L} sites.")
        
        # Sanity check matching dimensions
        print(f"Input site 0 shape:  {in_mps[0].data.shape}")
        print(f"Target site 0 shape: {out_mps[0].data.shape}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Testing the directory we just generated: N=8, T=3, mu=1.0, gamma=0.0
    data_dir = os.path.join(base_dir, "Learning_data", "N8_T3_mu1.0_gamma0.0")
    test_load_data(data_dir)
