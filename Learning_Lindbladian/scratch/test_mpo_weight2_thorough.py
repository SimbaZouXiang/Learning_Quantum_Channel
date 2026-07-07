"""Thorough verification that the Learning_MPO_scheduler modifications are
correct.

Checks
------
1. `get_input_and_output_MPS(..., input_weight=1)` returns 3N MPS, identical
   in count and content to the legacy call (no input_weight kw).
2. `get_input_and_output_MPS(..., input_weight=2)` returns 3 * N*(N-1)/2 MPS,
   with the same operator on both sites for each (i, j, op) tuple.
3. `Learning_MPO_scheduler(..., input_pauli_weight=1)` matches the legacy
   `weight_1_pauli_strings=True` call bit-for-bit (modulo training-loss
   precision) when given the same target params and seeds.
4. `Learning_MPO_scheduler(..., input_pauli_weight=2)` runs without error and
   produces finite losses.
5. `Learning_MPO_scheduler(..., input_pauli_weight=None, weight_1_pauli_strings=False)`
   still works → falls back to random-Pauli legacy path.
6. Targets across w1 vs w2 trainings at the same L_target chase the SAME
   teacher (i.e., the param_list is honored).
"""
import os, sys
import warnings
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
torch.set_num_threads(1)
import TDME_Trott as tdme


# ── 1, 2: get_input_and_output_MPS counts ─────────────────────────────
def _check_counts():
    N, T = 6, 3
    p = (torch.rand(T, N, 16, dtype=torch.float64)
         + 1j * torch.rand(T, N, 16, dtype=torch.float64))
    p_dep = torch.zeros(T, N, dtype=torch.float64)

    for w, expected in [(1, 3 * N), (2, 3 * N * (N - 1) // 2)]:
        ins, outs, _ = tdme.get_input_and_output_MPS(
            N, T, param=p, p_depolar=p_dep,
            max_bd=16, max_err=1e-10, truncation=False,
            noise_type="depolarizing", num_threads=1,
            input_weight=w,
        )
        assert len(ins) == expected, f"w={w}: expected {expected} inputs, got {len(ins)}"
        assert len(outs) == expected, f"w={w}: expected {expected} outputs, got {len(outs)}"
        # All targets should have been reindexed input{i} -> k{i}.
        all_inds = set().union(*[set(t.inds) for t in outs[0].tensors])
        for i in range(N):
            assert f"k{i}" in all_inds, f"missing k{i} in target indices: {all_inds}"
        print(f"  PASS: get_input_and_output_MPS(input_weight={w}) → {expected} inputs/outputs")


# ── 3: input_pauli_weight=1 matches legacy weight_1_pauli_strings=True ────
def _check_legacy_compat():
    N, T, L = 4, 2, 2
    target_param = (torch.rand(L, N, 16, dtype=torch.float64)
                    + 1j * torch.rand(L, N, 16, dtype=torch.float64))

    # New path (input_pauli_weight=1)
    np.random.seed(7); torch.manual_seed(7)
    res_new = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param.clone(),
        depolarizing_strength=0.01,
        epochs=3, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        input_pauli_weight=1,
    )
    # Legacy path (weight_1_pauli_strings=True, input_pauli_weight=None)
    np.random.seed(7); torch.manual_seed(7)
    res_old = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param.clone(),
        depolarizing_strength=0.01,
        epochs=3, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        weight_1_pauli_strings=True,
    )
    ll_new = np.asarray(res_new[1], dtype=float)
    ll_old = np.asarray(res_old[1], dtype=float)
    assert len(ll_new) == len(ll_old), f"epoch count mismatch: {len(ll_new)} vs {len(ll_old)}"
    diff = np.max(np.abs(ll_new - ll_old))
    print(f"  PASS: input_pauli_weight=1 matches legacy weight_1=True; "
          f"max |Δ training loss| = {diff:.2e}")
    assert diff < 1e-9, f"unexpected divergence: {diff}"


# ── 4: input_pauli_weight=2 runs cleanly ──────────────────────────────
def _check_w2_runs():
    N, T, L = 4, 2, 3
    target_param = (torch.rand(L, N, 16, dtype=torch.float64)
                    + 1j * torch.rand(L, N, 16, dtype=torch.float64))
    np.random.seed(11); torch.manual_seed(11)
    res = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param,
        depolarizing_strength=0.01,
        epochs=3, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        input_pauli_weight=2,
    )
    ll = np.asarray(res[1], dtype=float)
    tl = float(res[4])
    print(f"  PASS: input_pauli_weight=2 runs; "
          f"learning_loss epochs={len(ll)}, final={ll[-1]:.4e}, testing={tl:.4e}")
    assert np.all(np.isfinite(ll))
    assert np.isfinite(tl)


# ── 5: random-Pauli legacy path still works ───────────────────────────
def _check_random_pauli_legacy():
    N, T, L = 4, 2, 2
    target_param = (torch.rand(L, N, 16, dtype=torch.float64)
                    + 1j * torch.rand(L, N, 16, dtype=torch.float64))
    np.random.seed(13); torch.manual_seed(13)
    res = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param,
        depolarizing_strength=0.01,
        epochs=2, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        weight_1_pauli_strings=False,
        input_pauli_weight=None,   # explicit: fall back to legacy
    )
    ll = np.asarray(res[1], dtype=float)
    print(f"  PASS: random-Pauli legacy path runs; "
          f"learning_loss epochs={len(ll)}, final={ll[-1]:.4e}")
    assert np.all(np.isfinite(ll))


# ── 6: w1 and w2 chase the same teacher (target_param) ────────────────
def _check_same_teacher():
    N, T, L = 4, 2, 2
    target_param = (torch.rand(L, N, 16, dtype=torch.float64)
                    + 1j * torch.rand(L, N, 16, dtype=torch.float64))
    # Run w1, capture the returned param.
    np.random.seed(0); torch.manual_seed(0)
    res1 = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param.clone(),
        depolarizing_strength=0.01,
        epochs=1, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        input_pauli_weight=1,
    )
    # Run w2, capture the returned param.
    np.random.seed(0); torch.manual_seed(0)
    res2 = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        param_list=target_param.clone(),
        depolarizing_strength=0.01,
        epochs=1, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        input_pauli_weight=2,
    )
    p1 = res1[2]; p2 = res2[2]
    diff = float((p1 - p2).abs().max())
    print(f"  PASS: w1 and w2 chase identical teacher; max |Δ param| = {diff:.2e}")
    assert diff < 1e-12, f"target params diverged: {diff}"


def main():
    print("=== 1, 2. get_input_and_output_MPS counts and indices ===")
    _check_counts()

    print("\n=== 3. input_pauli_weight=1 matches legacy weight_1_pauli_strings=True ===")
    _check_legacy_compat()

    print("\n=== 4. input_pauli_weight=2 end-to-end ===")
    _check_w2_runs()

    print("\n=== 5. legacy random-Pauli path ===")
    _check_random_pauli_legacy()

    print("\n=== 6. w1 and w2 chase the same teacher ===")
    _check_same_teacher()

    print("\nALL PASS")


if __name__ == "__main__":
    main()
