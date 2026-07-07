import TDME_Trott as tdme
import torch
import gc
import sys

def get_mem():
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

# Just a quick check of tensor size
print("Dummy script complete.")
