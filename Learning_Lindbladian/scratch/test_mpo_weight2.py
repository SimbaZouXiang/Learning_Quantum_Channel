"""Smoke test for Learning_MPO_scheduler with input_pauli_weight={1,2}.

Verifies that the function runs end-to-end in both modes at a tiny problem
size (N=4, L=2) and produces finite training/testing losses.
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


def run_once(weight):
    np.random.seed(0); torch.manual_seed(0)
    N, T, L = 4, 2, 2
    res = tdme.Learning_MPO_scheduler(
        N=N, MPO_layer=T, model_to_learn_layer=L,
        depolarizing_strength=0.01,
        epochs=3, lr=0.05,
        normalized=False, max_bd=16, max_err=1e-6,
        truncation=False, noise_type="depolarizing",
        use_compressed=False, num_threads=1,
        input_pauli_weight=weight,
    )
    # 11-tuple: (model, learning_loss, param, p_depolar, testing_loss,
    #            testing_loss_list, params_np, p_depolar_np, p_dephaseX_np,
    #            p_dephaseY_np, p_dephaseZ_np)
    learning_loss = np.asarray(res[1], dtype=float)
    testing_loss = float(res[4])
    print(f"weight={weight}  epochs={len(learning_loss)}  "
          f"final_train={learning_loss[-1]:.4e}  testing={testing_loss:.4e}")
    assert np.all(np.isfinite(learning_loss))
    assert np.isfinite(testing_loss)


for w in (1, 2):
    run_once(w)
print("ALL PASS")
