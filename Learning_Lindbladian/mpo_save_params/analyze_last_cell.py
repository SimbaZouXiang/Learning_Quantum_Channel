"""Run only L=7, p=0.05 (the cell missing from the main analysis CSV)."""
import os, sys, warnings
warnings.filterwarnings("ignore", message="Casting complex")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np, torch
torch.set_num_threads(16)
import TDME_Trott as tdme

import analyze_mpo_distances as A  # reuses _load_student, _build_target_module, _compute_distance_on_inputs

L, p = 7, 0.05
print(f"=== L={L} p={p} ===", flush=True)
student_w1 = A._load_student(L, p, "w1")
student_combined = A._load_student(L, p, "combined")
target_module = A._build_target_module(L, p)

inputs_w1 = tdme.Pauli_MPS_weight_1(8)
inputs_w2 = tdme.Pauli_MPS_weight_2(8)
np.random.seed(1234); torch.manual_seed(1234)
inputs_rand = [tdme.random_pauli_MPS(8)[0] for _ in range(500)]

csv = os.path.join(SCRIPT_DIR, "figures", "mpo_pairwise_distances.csv")
with open(csv, "a") as f:
    for label, inputs in [("weight1_full", inputs_w1),
                          ("weight2_full", inputs_w2),
                          ("random_500",   inputs_rand)]:
        D1, D2, D3 = A._compute_distance_on_inputs(
            student_w1, student_combined, target_module, inputs,
        )
        f.write(f"{L},{p},{label},{D1:.6e},{D2:.6e},{D3:.6e}\n")
        f.flush()
        print(f"  {label:<15s}  ||w1-tgt||={D1:.4e}  "
              f"||cmb-tgt||={D2:.4e}  ||w1-cmb||={D3:.4e}", flush=True)
