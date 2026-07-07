import torch
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import get_random_input_output_MPS
try:
    param = torch.rand(2, 4, 16, dtype=torch.float64)+1j*torch.rand(2, 4, 16, dtype=torch.float64)
    p_depolar = torch.zeros((2, 4), dtype=torch.float64)
    get_random_input_output_MPS(N=4, T=2, param=param, p_depolar=p_depolar, num_threads=2)
    print("Success2")
except Exception as e:
    print(f"Error2: {e}")
