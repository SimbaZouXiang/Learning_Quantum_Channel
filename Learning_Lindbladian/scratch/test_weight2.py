"""Correctness sanity check for the new weight-2 Pauli training-input support.

Verifies:
  1. `Pauli_MPS_weight_2(N)` returns the right number of MPS, 3 * N*(N-1)/2,
     with X/Y/Z Pauli operators placed on exactly the right pair of sites
     and identity elsewhere.
  2. `get_input_and_output_MPS_TDME(..., input_weight=2)` routes to the new
     basis and produces an evolved target MPS for each input.
  3. `Learning_TDME_scheduler(..., input_pauli_weight=2)` runs end-to-end on
     a tiny (N=4, MPO_layer=2, model_to_learn_layer=2) problem for a few
     epochs without raising, and produces a finite training-loss curve.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch

torch.set_num_threads(1)
np.random.seed(0); torch.manual_seed(0)

import TDME_Trott as tdme


def _check_pauli_mps_weight_2(N):
    mps_list = tdme.Pauli_MPS_weight_2(N)
    expected = 3 * N * (N - 1) // 2
    assert len(mps_list) == expected, (
        f"Pauli_MPS_weight_2({N}): expected {expected} MPS, got {len(mps_list)}"
    )

    # Pauli basis convention is (I, X, Y, Z) -> physical-leg slots 0,1,2,3.
    OP_TO_SLOT = {"X": 1, "Y": 2, "Z": 3}
    pauli_ops = ["X", "Y", "Z"]

    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            for op in pauli_ops:
                M = mps_list[idx]
                # Each MPS site tensor has shape (left, 4, right) with all
                # bond dims = 1 in the bond_dim=1 input MPS.  Check that the
                # only non-zero physical-leg slot at site k matches the
                # expected operator (X/Y/Z at i,j; I elsewhere).
                for k in range(N):
                    data = M[k].data
                    # Squeeze bond dims of size 1 — works whether or not the
                    # tensor library returns them as size-1 axes.
                    vec = data.reshape(-1).numpy() if hasattr(data, "reshape") else np.asarray(data).reshape(-1)
                    if k in (i, j):
                        expected_slot = OP_TO_SLOT[op]
                    else:
                        expected_slot = 0   # identity component
                    # vec has length 4 (since bond dims are 1).
                    assert vec.shape[0] == 4, f"unexpected vec shape {vec.shape}"
                    nonzero = np.flatnonzero(np.abs(vec) > 1e-12)
                    assert nonzero.tolist() == [expected_slot], (
                        f"({i},{j},{op}) site {k}: expected slot {expected_slot}, "
                        f"got non-zero slots {nonzero.tolist()} (vec={vec})"
                    )
                idx += 1
    print(f"  PASS: Pauli_MPS_weight_2(N={N}) → {expected} MPS, all correct")


def _check_get_input_and_output():
    N = 4
    T = 2
    mu = 1.0
    gamma = [0.05] * N
    J = 1.0
    t = 0.3
    inputs, targets, _ = tdme.get_input_and_output_MPS_TDME(
        N, T, mu=mu, gamma=gamma, J=J, t=t,
        max_bd=16, max_err=1e-10, truncation=False, num_threads=1,
        input_weight=2,
    )
    expected = 3 * N * (N - 1) // 2
    assert len(inputs) == expected, f"expected {expected} inputs, got {len(inputs)}"
    assert len(targets) == expected, f"expected {expected} targets, got {len(targets)}"
    # Targets should have been reindexed input{i} → k{i}.
    sample_target = targets[0]
    target_inds = set().union(*[set(t.inds) for t in sample_target.tensors])
    for i in range(N):
        assert f"k{i}" in target_inds, f"missing k{i} in target indices: {target_inds}"
    print(f"  PASS: get_input_and_output_MPS_TDME(..., input_weight=2) → "
          f"{expected} inputs + {expected} targets")


def _check_scheduler():
    N = 4
    MPO_layer = 2
    L = 2
    mu = 1.0
    gamma = [0.05] * N
    J = 1.0
    t = 0.3
    print("  Running Learning_TDME_scheduler with input_pauli_weight=2 ...")
    result = tdme.Learning_TDME_scheduler(
        N=N, MPO_layer=MPO_layer, model_to_learn_layer=L,
        mu=mu, gamma=gamma, J=J, t=t,
        epochs=3, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-10,
        truncation=False, noise_type="dephasing",
        use_scheduler=False, use_compressed=False,
        num_threads=1, data_dir=None,
        input_pauli_weight=2,
    )
    # Return tuple: (model, learning_loss, testing_loss, testing_loss_list)
    model, learning_loss, testing_loss, testing_loss_list = result
    arr = np.asarray(learning_loss, dtype=float)
    print(f"  learning_loss curve length = {len(arr)}; final = {arr[-1]:.5e}")
    print(f"  testing_loss = {float(testing_loss):.5e}")
    assert len(arr) >= 3 + (3 // 5), "learning_loss curve shorter than expected"
    assert np.all(np.isfinite(arr)), f"non-finite values in learning_loss: {arr}"
    assert np.isfinite(testing_loss), f"non-finite testing_loss: {testing_loss}"
    print("  PASS: end-to-end scheduler runs with input_pauli_weight=2")


def main():
    print("=== Pauli_MPS_weight_2 count + content ===")
    for n in (3, 4, 5, 8):
        _check_pauli_mps_weight_2(n)

    print("\n=== get_input_and_output_MPS_TDME with input_weight=2 ===")
    _check_get_input_and_output()

    print("\n=== Learning_TDME_scheduler with input_pauli_weight=2 ===")
    _check_scheduler()

    print("\nALL PASS")


if __name__ == "__main__":
    main()
