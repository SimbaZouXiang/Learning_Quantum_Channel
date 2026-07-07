import time
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from TDME_Trott import get_input_and_output_MPS_TDME
import torch

def test_speed():
    # Preventing PyTorch threads from clashing with ProcessPool workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    N = 10
    T = 3
    mu = 1.0
    gamma = [0.1] * N
    
    print(f"Testing N={N}, T={T} Sequential...")
    t0 = time.time()
    get_input_and_output_MPS_TDME(N=N, T=T, mu=mu, gamma=gamma, num_threads=1)
    t_seq = time.time() - t0
    print(f"Sequential Time: {t_seq:.2f}s")
    
    print(f"\nTesting N={N}, T={T} Parallel... (4 workers)")
    t0 = time.time()
    get_input_and_output_MPS_TDME(N=N, T=T, mu=mu, gamma=gamma, num_threads=4)
    t_par = time.time() - t0
    print(f"Parallel Time: {t_par:.2f}s")
    
    print(f"\nSpeedup: {t_seq / t_par:.2f}x")

if __name__ == "__main__":
    test_speed()
