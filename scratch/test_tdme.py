import torch
import sys
import os
from TDME_Trott import get_input_and_output_MPS_TDME

def main():
    print("Running sequential test...")
    inputs, outputs, err = get_input_and_output_MPS_TDME(
        N=4, T=2, mu=1.0, gamma=[0.1]*4, J=1.0, t=0.1, 
        num_threads=1
    )
    print(f"[Sequential] Generated {len(inputs)} inputs and {len(outputs)} outputs.\n")

    print("Running parallel test with 2 workers...")
    p_inputs, p_outputs, p_err = get_input_and_output_MPS_TDME(
        N=4, T=2, mu=1.0, gamma=[0.1]*4, J=1.0, t=0.1, 
        num_threads=2
    )
    print(f"[Parallel] Generated {len(p_inputs)} inputs and {len(p_outputs)} outputs.\n")

    assert len(inputs) == 12, f"Expected 12 inputs, got {len(inputs)}"
    assert len(inputs) == len(p_inputs), "Sequential and parallel input length mismatch!"
    assert len(outputs) == len(p_outputs), "Sequential and parallel output length mismatch!"
    
    # Calculate distance to ensure identical results
    from quimb.tensor.fitting import tensor_network_distance
    diff_sum = 0
    for o1, o2 in zip(outputs, p_outputs):
        diff_sum += tensor_network_distance(o1, o2).item()
        
    print(f"Total distance between sequential and parallel outputs: {diff_sum}")
    if diff_sum < 1e-10:
        print("SUCCESS! Sequential and parallel results match.")
    else:
        print("WARNING: Divergence between sequential and parallel results.")

if __name__ == "__main__":
    main()
