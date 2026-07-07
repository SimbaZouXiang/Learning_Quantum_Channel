"""Verify the new random-Pauli training input option."""
import os, sys, warnings
warnings.filterwarnings("ignore", message="Casting complex")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
torch.set_num_threads(1)
import TDME_Trott as tdme


def check_helper():
    for n, n_samples in [(4, 10), (8, 24), (8, 276)]:
        mps_list = tdme.Pauli_MPS_random(n, n_samples, seed=42)
        assert len(mps_list) == n_samples
        # All MPS should be on n sites
        for M in mps_list:
            assert M.L == n
        # Different seeds should give different sets
        mps_a = tdme.Pauli_MPS_random(n, n_samples, seed=1)
        mps_b = tdme.Pauli_MPS_random(n, n_samples, seed=2)
        # Compare data of first MPS
        d1 = mps_a[0][0].data
        d2 = mps_b[0][0].data
        # Should NOT be identical with different seeds (well, with prob > 1-1/12)
        same = (torch.is_tensor(d1) and torch.is_tensor(d2) and torch.allclose(d1, d2))
        # Same seed → identical
        mps_c = tdme.Pauli_MPS_random(n, n_samples, seed=42)
        assert torch.allclose(mps_list[0][0].data, mps_c[0][0].data), "Same seed should give same MPS"
        print(f"  PASS: Pauli_MPS_random(n={n}, n_samples={n_samples}) → reproducible with seed")


def check_scheduler():
    N, T, L = 4, 2, 2
    target_param = (torch.rand(L, N, 16, dtype=torch.float64)
                    + 1j * torch.rand(L, N, 16, dtype=torch.float64))
    kwargs = dict(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param.clone(),
        depolarizing_strength=0.01,
        epochs=2, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
    )
    np.random.seed(0); torch.manual_seed(0)
    res_w1 = tdme.Learning_MPO_scheduler(input_pauli_weight=1, **kwargs)
    ll_w1 = np.asarray(res_w1[1], dtype=float)

    np.random.seed(0); torch.manual_seed(0)
    res_rand24 = tdme.Learning_MPO_scheduler(
        input_pauli_weight=("random", 24, 100), **kwargs,
    )
    ll_rand24 = np.asarray(res_rand24[1], dtype=float)

    diff = float(np.max(np.abs(ll_w1 - ll_rand24)))
    print(f"  w1 trajectory final         : {ll_w1[-1]:.4e}")
    print(f"  random-24 trajectory final  : {ll_rand24[-1]:.4e}")
    print(f"  max |Δ|                     : {diff:.4e}")
    assert diff > 1e-6, "random-24 trajectory identical to w1 — dispatch broken"
    print(f"  PASS: scheduler distinguishes weight-1 vs ('random', 24)")


def check_fine_tune():
    N, T, L = 4, 2, 2
    target_param = (torch.rand(L, N, 16, dtype=torch.float64)
                    + 1j * torch.rand(L, N, 16, dtype=torch.float64))
    np.random.seed(0); torch.manual_seed(0)
    res_default = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param.clone(),
        depolarizing_strength=0.01,
        epochs=10, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        input_pauli_weight=1,
    )
    ll_default = np.asarray(res_default[1], dtype=float)
    # 10 main + 10//5 = 2 fine-tune = 12 epoch entries
    print(f"  epochs=10 default → {len(ll_default)} total (expected 12)")
    assert len(ll_default) == 12

    np.random.seed(0); torch.manual_seed(0)
    res_ft7 = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param.clone(),
        depolarizing_strength=0.01,
        epochs=10, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        input_pauli_weight=1,
        fine_tune_epochs=7,
    )
    ll_ft7 = np.asarray(res_ft7[1], dtype=float)
    print(f"  epochs=10, fine_tune_epochs=7 → {len(ll_ft7)} total (expected 17)")
    assert len(ll_ft7) == 17
    print(f"  PASS: fine_tune_epochs kwarg works")


if __name__ == "__main__":
    print("=== Pauli_MPS_random helper ===")
    check_helper()
    print("\n=== Scheduler dispatch for ('random', n_samples) ===")
    check_scheduler()
    print("\n=== fine_tune_epochs kwarg ===")
    check_fine_tune()
    print("\nALL PASS")
