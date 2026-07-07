"""Diagnostic for the weight-2 testing-loss anomaly.

After training on weight-2 inputs at L_target = MPO_layer = 3 (the easiest
case, where the student is at least as expressive as the target), we evaluate
the *same* trained model on three different input sets:
  (a) the weight-2 training set itself,
  (b) the full weight-1 Pauli basis (24 inputs at N=8),
  (c) 30 random Pauli MPS inputs (uniform weight ∈ [1, N]).

If the model genuinely learned the channel, all three should give similar
loss. A large discrepancy points at either an overfitting/generalization
mechanism or a bug.

In addition, we print the noise channel rates the optimizer settled on, to
check whether the model has saturated the clamp (which would explain
underfitting on weight-2 vs perfect-looking weight-1 fits).
"""
import os, sys, time
import warnings
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
torch.set_num_threads(8)
import TDME_Trott as tdme

# Fixed reproducible seed.
np.random.seed(42); torch.manual_seed(42)

N      = 8
T      = 3
L      = 3                # critical case: matched depth
mu     = 1
J      = 1
gamma  = [0.1] * N
t_val  = 1.0
EPOCHS = 200              # mirrors w2_e200 in the user's run

print(f"=== Train PQC at N={N}, T={T}, L={L} with weight-2 inputs ({EPOCHS} epochs) ===",
      flush=True)
t0 = time.time()
res = tdme.Learning_TDME_scheduler(
    N=N, MPO_layer=T, model_to_learn_layer=L,
    mu=mu, gamma=gamma, J=J, t=t_val,
    epochs=EPOCHS, lr=0.05,
    normalized=False, max_bd=64, max_err=1e-8,
    truncation=True, noise_type="dephasing",
    use_scheduler=False, use_compressed=False,
    num_threads=1, data_dir=None,
    input_pauli_weight=2,
)
elapsed = time.time() - t0
model, learning_loss, testing_loss, testing_loss_list = res
print(f"  Trained in {elapsed:.1f}s. final_train_loss = {learning_loss[-1]:.4e}  "
      f"reported_test_loss = {float(testing_loss):.4e}", flush=True)

# ── Re-evaluate the trained model on three different input distributions ─
def _avg_loss_over(input_mps_list, label):
    """Average Frobenius distance over a list of input MPS."""
    losses = []
    with torch.no_grad():
        mpo_fit = model.get_MPO(noise_type="dephasing")
        # Get the target via the same TDME pipeline.
        r = t_val / L
        all_u = tdme.construct_TDME_unitary(N, L, r=r, mu=mu, J=J)
        all_j = tdme.construct_jump_matrices(N, gamma, r=r)
        for inp in input_mps_list:
            tgt, _ = tdme.Pauli_MPS_after_TDME_output_only(
                inp.copy(), L, r=r,
                all_unitary=all_u, all_jumping=all_j,
                max_bd=64, max_err=1e-8, truncation=True,
            )
            for i, ten in enumerate(tgt):
                ten.reindex_({f'input{i}': f'k{i}'})
            student_out = mpo_fit | inp
            loss = tdme.tensor_network_distance(
                student_out.astype("complex128"),
                tgt.astype("complex128"),
            ).item()
            losses.append(loss)
    arr = np.asarray(losses, dtype=float)
    print(f"  {label:<18s}  n={len(arr):3d}  mean={arr.mean():.4e}  "
          f"median={np.median(arr):.4e}  min={arr.min():.4e}  max={arr.max():.4e}",
          flush=True)
    return arr

print("\n=== Re-evaluate trained model on three input distributions ===")
arr_w1 = _avg_loss_over(tdme.Pauli_MPS_weight_1(N),  "weight-1 (24)")
arr_w2 = _avg_loss_over(tdme.Pauli_MPS_weight_2(N),  "weight-2 (84)")
np.random.seed(99)
rand_inputs = [tdme.random_pauli_MPS(N)[0] for _ in range(30)]
arr_rd = _avg_loss_over(rand_inputs, "random (30)")

# ── Inspect learned noise rates and gate parameters ─────────────────────
print("\n=== Trained noise rates (per layer, per site) ===")
for name, tens in [("p_depolar", model.p_depolar),
                    ("p_dephaseX", model.p_dephaseX),
                    ("p_dephaseY", model.p_dephaseY),
                    ("p_dephaseZ", model.p_dephaseZ)]:
    arr = tens.detach().numpy()
    print(f"  {name:>11s}  shape={arr.shape}  "
          f"mean={arr.mean():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")

print("\n=== Headline ratios ===")
print(f"  train_loss(weight-2 set)  = {arr_w2.mean():.4e}")
print(f"  test_loss(weight-1 set)   = {arr_w1.mean():.4e}")
print(f"  test_loss(random set)     = {arr_rd.mean():.4e}")
print(f"  ratio w1/w2  = {arr_w1.mean()/arr_w2.mean():.2f}")
print(f"  ratio rd/w2  = {arr_rd.mean()/arr_w2.mean():.2f}")
