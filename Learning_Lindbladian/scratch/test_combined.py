"""Verify the new 'combined' (weight-1 + weight-2-full) Pauli input set."""
import os, sys, warnings
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
torch.set_num_threads(1)
import TDME_Trott as tdme


def check_count():
    for N in (3, 4, 5, 8):
        combined = tdme.Pauli_MPS_combined_1_and_2_full(N)
        expected = 3 * N + 9 * N * (N - 1) // 2
        assert len(combined) == expected, (
            f"N={N}: expected {expected}, got {len(combined)}"
        )
        # Top 3N entries should match Pauli_MPS_weight_1; the rest should
        # match Pauli_MPS_weight_2_full. We don't compare object identity
        # but the data fingerprints.
        w1 = tdme.Pauli_MPS_weight_1(N)
        w2f = tdme.Pauli_MPS_weight_2_full(N)
        assert len(w1) == 3 * N
        assert len(w2f) == 9 * N * (N - 1) // 2
        assert len(combined) == len(w1) + len(w2f)
        for i, (a, b) in enumerate(zip(combined[: len(w1)], w1)):
            for k in range(N):
                va = a[k].data.reshape(-1)
                vb = b[k].data.reshape(-1)
                assert torch.allclose(va, vb), f"N={N} idx={i} site={k}: weight-1 prefix mismatch"
        for i, (a, b) in enumerate(zip(combined[len(w1):], w2f)):
            for k in range(N):
                va = a[k].data.reshape(-1)
                vb = b[k].data.reshape(-1)
                assert torch.allclose(va, vb), f"N={N} idx={i} site={k}: weight-2-full suffix mismatch"
        print(f"  PASS: N={N} combined count={expected} (= 3N + 9N(N-1)/2); ordering correct")


def check_scheduler_path():
    """Train the same problem twice — once on 'combined', once on weight-1 —
    and confirm trajectories differ (rules out fall-through to weight-1)."""
    N = 4
    target_param = (torch.rand(2, N, 16, dtype=torch.float64)
                    + 1j * torch.rand(2, N, 16, dtype=torch.float64))
    kwargs = dict(
        N=N, MPO_layer=2, model_to_learn_layer=2,
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
    res_c = tdme.Learning_MPO_scheduler(input_pauli_weight="combined", **kwargs)
    ll_c = np.asarray(res_c[1], dtype=float)

    diff = float(np.max(np.abs(ll_w1 - ll_c)))
    print(f"  weight-1 trajectory final: {ll_w1[-1]:.4e}")
    print(f"  combined trajectory final: {ll_c[-1]:.4e}")
    print(f"  max |Δ|: {diff:.4e}")
    assert diff > 1e-6, "combined and weight-1 trajectories are bit-identical — dispatch is broken"
    print(f"  PASS: Learning_MPO_scheduler distinguishes weight-1 vs combined")


if __name__ == "__main__":
    print("=== Combined input-set generation ===")
    check_count()
    print("\n=== Scheduler dispatch ===")
    check_scheduler_path()
    print("\nALL PASS")
