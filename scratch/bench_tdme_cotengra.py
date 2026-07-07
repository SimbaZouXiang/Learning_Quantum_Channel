import time
import torch
import quimb.tensor as qtn
import cotengra as ctg

N = 8
chi = 64
chi_mpo = 16

print(f"Benchmarking <MPS | MPO | MPS> structurally identical to TDME_Trott")

# Let's manually construct M = mpo | mps and M_tar = mps
mps_in = qtn.MPS_rand_state(N, bond_dim=chi, tags='mps_in')
mpo = qtn.MPO_rand_herm(N, bond_dim=chi_mpo, tags='mpo')
mps_target = qtn.MPS_rand_state(N, bond_dim=chi, tags='mps_tar')

# Re-index MPO and input MPS to make them a single TN representing the evolved state
for i in range(N):
    mps_in[i].reindex_({f'k{i}': f'input{i}'})
    
m_evolved = mps_in | mpo

print("\n--- Warmup Cotengra ---")
opt = ctg.ReusableHyperOptimizer(methods=['greedy'], max_repeats=32, parallel=False)

def check_dist(opt_str):
    M_e = m_evolved.copy()
    M_t = mps_target.copy()
    M_e.add_tag('bra')
    M_t.add_tag('ket')
    # Reindex target
    for i in range(N):
        M_t[i].reindex_({f'k{i}': f'k{i}_b'})
    
    overlap_tn = M_e | M_t
    return overlap_tn.contract(optimize=opt_str)


# Measure single
t0 = time.time()
_ = check_dist(opt)
t_ctg_p = time.time() - t0
print(f"Warmed up Cotengra: {t_ctg_p:.4f} s")

# Measure Cached Cotengra
t0 = time.time()
for _ in range(5):
    _ = check_dist(opt)
t_ctg = (time.time() - t0) / 5
print(f"Average 'cotengra' (cached) time: {t_ctg:.4f} s/iter")

# Measure Auto
t0 = time.time()
try:
    _ = check_dist('auto')
    t_auto = time.time() - t0
    print(f"Average 'auto' time (1 iter):     {t_auto:.4f} s/iter")
    if t_ctg > 0:
        print(f"Speedup factor (Cached vs Auto):  {t_auto/t_ctg:.2f}x")
except Exception as e:
    print(f"Auto crashed: {e}")

