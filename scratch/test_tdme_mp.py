import torch
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import get_input_and_output_MPS_TDME

if __name__ == "__main__":
    try:
        inputs, outputs, err = get_input_and_output_MPS_TDME(
            N=30, T=3, mu=1.0, gamma=[0.1]*30, J=1.0, t=1.0, 
            num_threads=2
        )
        print(f"Success! {len(inputs)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
