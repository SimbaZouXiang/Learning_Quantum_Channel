import time
import quimb.tensor as qtn
import cotengra as ctg

opt = ctg.ReusableHyperOptimizer(methods=['greedy'], max_repeats=16, max_time=0.5, parallel=False, progbar=False)

def build_test_network(N, use_uuid):
    mps1 = qtn.MPS_rand_state(N, bond_dim=4, tags='m1')
    mpo = qtn.MPO_rand(N, bond_dim=4, tags='mpo')
    for i in range(N):
        mpo[i].reindex_({f'k{i}': f'mid{i}' if not use_uuid else qtn.rand_uuid()})
        mps1[i].reindex_({f'k{i}': f'mid{i}' if not use_uuid else qtn.rand_uuid()})
    return mpo | mps1, qtn.MPS_rand_state(N, bond_dim=4, tags='m2')

def test_cache(use_uuid=False):
    t0 = time.time()
    for _ in range(10):
        tn1, mps2 = build_test_network(8, use_uuid)
        qtn.tensor_network_distance(tn1, mps2, optimize=opt)
    elapsed = time.time() - t0
    print(f"[{'UUID' if use_uuid else 'FIXED'}] Elapsed for 10 loops: {elapsed:.3f} s")

test_cache(use_uuid=True)  # Populates useless caches
test_cache(use_uuid=False) # Populates good cache
test_cache(use_uuid=False) # Hits cache heavily!
test_cache(use_uuid=True)  # Misses cache!

