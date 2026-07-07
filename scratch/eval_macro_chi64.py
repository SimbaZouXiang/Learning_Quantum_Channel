import time
import sys
import torch
import quimb.tensor as qtn

sys.path.append("/scratch/simba/")
from Learning_Lindbladian import TDME_Trott

def run_bench(name):
    print(f"\n--- Running Training Epoch with {name} ---")
    t0 = time.time()
    
    # 1 epoch, reduced threads to isolate TDME operations cleaner
    TDME_Trott.Learning_TDME_scheduler(
        N=8, MPO_layer=2, model_to_learn_layer=2, mu=0.5, gamma=[0.1]*8, J=1.0, t=0.6,
        epochs=1, lr=0.01, normalized=False, max_bd=64, max_err=1E-6, 
        truncation=True, noise_type="all", use_scheduler=False,
        use_compressed=True, num_threads=4 
    )
    
    elapsed = time.time() - t0
    print(f"{name} Complete: {elapsed:.2f} seconds")
    return elapsed

# 1) Run with the current patch (Cotengra)
t_ctg = run_bench("Cotengra Optimizer (Patched)")

# 2) Unpatch dynamically to revert to default 'opt-einsum' (auto)
from quimb.tensor.fitting import tensor_network_distance as orig_tnd
TDME_Trott.tensor_network_distance = orig_tnd

t_auto = run_bench("Original Opt-Einsum (Auto)")

print("\n--- Summary (N=8, max_bd=64) ---")
print(f"Cotengra:   {t_ctg:.2f} s")
print(f"Opt-Einsum: {t_auto:.2f} s")
if t_ctg < t_auto:
    print(f"Speedup: Cotengra is {t_auto/t_ctg:.2f}x faster.")
else:
    print(f"Speedup: Auto is {t_ctg/t_auto:.2f}x faster.")

