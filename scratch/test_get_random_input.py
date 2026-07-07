import torch
from TDME_Trott import get_random_input_output_MPS

N = 4
T = 2

# generate simple param and depolarizing
param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
p_depolar = torch.zeros((int(T), int(N)), dtype=torch.float64)

# Test sequentially
print("Testing sequential...")
mps_list_seq, target_mps_list_seq, param_seq = get_random_input_output_MPS(
    N, T, param, p_depolar, no_sample=2, p_dephaseX=None, p_dephaseY=None, p_dephaseZ=None,
    truncation=False, max_bd=64, max_err=1E-10, noise_type="depolarizing", num_threads=1
)
print("Sequential success, generated", len(target_mps_list_seq), "samples.")

# Test parallel
print("Testing parallel (num_threads=2)...")
mps_list_par, target_mps_list_par, param_par = get_random_input_output_MPS(
    N, T, param, p_depolar, no_sample=2, p_dephaseX=None, p_dephaseY=None, p_dephaseZ=None,
    truncation=False, max_bd=64, max_err=1E-10, noise_type="depolarizing", num_threads=2
)
print("Parallel success, generated", len(target_mps_list_par), "samples.")

print("All tests passed!")
