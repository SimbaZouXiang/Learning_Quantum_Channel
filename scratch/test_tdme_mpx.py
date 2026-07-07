import torch
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import get_input_and_output_MPS_TDME

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    try:
        inputs, outputs, err = get_input_and_output_MPS_TDME(
            N=10, T=3, mu=1.0, gamma=[0.1]*10, J=1.0, t=1.0, 
            num_threads=4
        )
        print(f"Success! Generated {len(inputs)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
