"""Verify Pauli_MPS_weight_2_full(N) generates the correct 9 * N(N-1)/2 set.

Checks:
  (1) Count matches 9 * N*(N-1)/2 for N in {3, 4, 5, 8}.
  (2) For every (i, j, op1, op2) tuple there is an MPS whose only non-identity
      slot at sites i and j is exactly the requested operator, and identity
      everywhere else.
  (3) `get_input_and_output_MPS(..., input_weight="2_full")` routes to it.
  (4) `Learning_MPO_scheduler(..., input_pauli_weight="2_full")` runs e2e.
"""
import os, sys, warnings
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import itertools
import numpy as np
import torch
torch.set_num_threads(1)
import TDME_Trott as tdme

OP_TO_SLOT = {"X": 1, "Y": 2, "Z": 3}
pauli_ops = ["X", "Y", "Z"]


def check_count_and_content():
    for N in (3, 4, 5, 8):
        mps_list = tdme.Pauli_MPS_weight_2_full(N)
        expected = 9 * N * (N - 1) // 2
        assert len(mps_list) == expected, f"N={N}: expected {expected}, got {len(mps_list)}"
        # Walk every (i, j, op1, op2) in lex order; check the MPS at that index
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                for op1, op2 in itertools.product(pauli_ops, repeat=2):
                    M = mps_list[idx]
                    expected_slots = {i: OP_TO_SLOT[op1], j: OP_TO_SLOT[op2]}
                    for site in range(N):
                        vec = M[site].data.reshape(-1)
                        if torch.is_tensor(vec):
                            vec = vec.numpy()
                        nonzero = np.flatnonzero(np.abs(vec) > 1e-12).tolist()
                        want = expected_slots.get(site, 0)
                        assert nonzero == [want], (
                            f"N={N} idx={idx} (i,j,op1,op2)=({i},{j},{op1},{op2}) "
                            f"site={site}: expected slot {want}, got {nonzero}"
                        )
                    idx += 1
        print(f"  PASS: N={N}  count={expected}  all (i,j,op1,op2) tuples correct")


def check_pipeline_endpoints():
    # get_input_and_output_MPS route
    N, T = 5, 2
    p = (torch.rand(T, N, 16, dtype=torch.float64)
         + 1j * torch.rand(T, N, 16, dtype=torch.float64))
    p_dep = torch.zeros(T, N, dtype=torch.float64)
    ins, outs, _ = tdme.get_input_and_output_MPS(
        N, T, param=p, p_depolar=p_dep,
        max_bd=16, max_err=1e-10, truncation=False,
        noise_type="depolarizing", num_threads=1,
        input_weight="2_full",
    )
    expected = 9 * N * (N - 1) // 2
    assert len(ins) == expected, f"got {len(ins)}, expected {expected}"
    assert len(outs) == expected
    print(f"  PASS: get_input_and_output_MPS(input_weight='2_full') → {expected} pairs")

    # Learning_MPO_scheduler route — distinguish weight=1 vs weight=2_full by
    # checking that the two paths produce DIFFERENT training-loss trajectories.
    # (The earlier "smoke-only" test would pass even when the function silently
    # fell back to weight-1; this one catches that regression.)
    np.random.seed(0); torch.manual_seed(0)
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
    res1 = tdme.Learning_MPO_scheduler(input_pauli_weight=1, **kwargs)
    ll1 = np.asarray(res1[1], dtype=float)

    np.random.seed(0); torch.manual_seed(0)
    res_full = tdme.Learning_MPO_scheduler(input_pauli_weight="2_full", **kwargs)
    ll_full = np.asarray(res_full[1], dtype=float)

    # The two training trajectories should differ (different training sets →
    # different per-epoch mean losses).
    diff = float(np.max(np.abs(ll1 - ll_full)))
    print(f"  weight-1 trajectory final: {ll1[-1]:.4e}")
    print(f"  weight-2-full trajectory final: {ll_full[-1]:.4e}")
    print(f"  max |Δ| between trajectories: {diff:.4e}")
    assert np.all(np.isfinite(ll_full))
    assert diff > 1e-6, (
        f"weight-1 and 2_full trajectories are bit-identical — scheduler "
        f"likely fell through to weight-1 path; check input_pauli_weight dispatch."
    )
    print(f"  PASS: Learning_MPO_scheduler distinguishes weight-1 vs 2_full")


if __name__ == "__main__":
    print("=== Pauli_MPS_weight_2_full count and content ===")
    check_count_and_content()
    print("\n=== Plumbing through scheduler ===")
    check_pipeline_endpoints()
    print("\nALL PASS")
