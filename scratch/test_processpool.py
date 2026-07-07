import torch
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import get_input_and_output_MPS_TDME
try:
    get_input_and_output_MPS_TDME(N=4, T=2, mu=1.0, gamma=[0.1]*4, num_threads=2)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
