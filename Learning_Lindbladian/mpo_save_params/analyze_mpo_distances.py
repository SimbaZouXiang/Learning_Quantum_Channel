"""After save-params job (1582956) finishes, reconstruct the trained student
MPOs and compute three pairwise distances per cell:

  D₁ = || M_w1_student        - M_target ||  (channel-difference Frobenius)
  D₂ = || M_combined_student  - M_target ||  (same, combined training)
  D₃ = || M_w1_student        - M_combined_student ||  (how far apart are the two students?)

Distance is computed by averaging the per-sample squared Frobenius distance
over three input sets and reporting all three:

  (a) full weight-1 Pauli basis           24 inputs (complete for weight 1)
  (b) full weight-2 Pauli basis (XX..ZZ)  252 inputs (complete for weight 2)
  (c) 500 uniform-weight random Paulis    (Monte Carlo over the full 4^N basis)

Outputs a CSV with columns:
  L, p, eval_set, D1, D2, D3
"""
import os, sys
import warnings
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch
torch.set_num_threads(8)
import TDME_Trott as tdme

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]
MAX_BD     = 32
MAX_ERR    = 1e-6
NOISE_TYPE = "depolarizing"
TRUNCATION = False
N_RANDOM   = 500

RD = os.path.join(SCRIPT_DIR, "results")
FD = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FD, exist_ok=True)


def _load_student(L, p, variant_tag):
    """Reconstruct a trained student QMLM from saved .npy arrays.

    state_dict alone is unreliable here: the QMLM __init__ has a branch
    (L_target == T_model) where `self.params = param` bypasses nn.Parameter,
    so the saved state_dict for L=3 cells is missing the 'params' key.
    We instead load all four parameter arrays from .npy and inject directly.
    """
    prefix = os.path.join(RD, f"N{N}_T{T}_L{L}_p{p}_{variant_tag}")
    params_np    = np.load(prefix + "_params.npy",   allow_pickle=True)
    p_depolar    = np.load(prefix + "_p_depolar.npy", allow_pickle=True)
    p_dephaseX   = np.load(prefix + "_p_dephaseX.npy", allow_pickle=True)
    p_dephaseY   = np.load(prefix + "_p_dephaseY.npy", allow_pickle=True)
    p_dephaseZ   = np.load(prefix + "_p_dephaseZ.npy", allow_pickle=True)

    model = tdme.QMLM(N, T, max_bd=MAX_BD, max_err=MAX_ERR)
    # Overwrite the default identity init with the trained values.
    model.params = torch.nn.Parameter(
        torch.from_numpy(params_np).to(torch.complex128), requires_grad=False,
    )
    model.p_depolar.data    = torch.from_numpy(p_depolar).to(torch.float64)
    model.p_dephaseX.data   = torch.from_numpy(p_dephaseX).to(torch.float64)
    model.p_dephaseY.data   = torch.from_numpy(p_dephaseY).to(torch.float64)
    model.p_dephaseZ.data   = torch.from_numpy(p_dephaseZ).to(torch.float64)
    return model


def _build_target_module(L, p):
    """Reconstruct the target QMLM_output_only with deterministic teacher params."""
    target_param = torch.load(
        os.path.join(RD, f"N{N}_T{T}_L{L}_p{p}_w1_target_param.pt"),
        map_location="cpu",
        weights_only=True,
    )
    p_depolar_tgt = torch.ones((L, N), dtype=torch.float64) * p
    return tdme.QMLM_output_only(
        N, L, param=target_param, p_depolar=p_depolar_tgt,
        max_bd=MAX_BD, max_err=MAX_ERR,
    )


def _compute_distance_on_inputs(student_a, student_b, target_module, input_mps_list):
    """Average squared Frobenius distance over the input set, for three pairs.

    student_a, student_b: QMLM modules (build MPO via get_MPO).
    target_module: QMLM_output_only (apply via forward_depolarizing_only).
    Returns (D1_mean, D2_mean, D3_mean) where:
        D1 = ||student_a(P) - target(P)||^2
        D2 = ||student_b(P) - target(P)||^2
        D3 = ||student_a(P) - student_b(P)||^2
    """
    D1, D2, D3 = [], [], []
    with torch.no_grad():
        mpo_a = student_a.get_MPO(noise_type=NOISE_TYPE)
        mpo_b = student_b.get_MPO(noise_type=NOISE_TYPE)
        for inp in input_mps_list:
            tgt = target_module.forward_depolarizing_only(
                inp.copy(), truncation=TRUNCATION,
            )
            for i, ten in enumerate(tgt):
                ten.reindex_({f'input{i}': f'k{i}'})
            out_a = mpo_a | inp
            out_b = mpo_b | inp
            tgt_c = tgt.astype("complex128")
            out_a_c = out_a.astype("complex128")
            out_b_c = out_b.astype("complex128")
            D1.append(tdme.tensor_network_distance(out_a_c, tgt_c).item())
            D2.append(tdme.tensor_network_distance(out_b_c, tgt_c).item())
            D3.append(tdme.tensor_network_distance(out_a_c, out_b_c).item())
    return float(np.mean(D1)), float(np.mean(D2)), float(np.mean(D3))


def main():
    rows = []
    # Pre-compute the input sets that are basis-complete.
    inputs_w1 = tdme.Pauli_MPS_weight_1(N)
    inputs_w2 = tdme.Pauli_MPS_weight_2(N)
    # Reproducible random Pauli set
    np.random.seed(1234)
    torch.manual_seed(1234)
    inputs_rand = [tdme.random_pauli_MPS(N)[0] for _ in range(N_RANDOM)]

    eval_sets = [
        ("weight1_full", inputs_w1),
        ("weight2_full", inputs_w2),
        (f"random_{N_RANDOM}", inputs_rand),
    ]

    csv = os.path.join(FD, "mpo_pairwise_distances.csv")
    # Open the CSV up-front and flush each row so partial results survive
    # if the wall-clock runs out before the final cell finishes.
    with open(csv, "w") as f:
        f.write("L,p,eval_set,D_w1_target,D_combined_target,D_w1_combined\n")
        f.flush()
        for L in LTARGETS:
            for p in DEPOL_LIST:
                student_w1 = _load_student(L, p, "w1")
                student_combined = _load_student(L, p, "combined")
                target_module = _build_target_module(L, p)
                print(f"\n=== L={L} p={p} ===", flush=True)
                for label, inputs in eval_sets:
                    D1, D2, D3 = _compute_distance_on_inputs(
                        student_w1, student_combined, target_module, inputs,
                    )
                    rows.append([L, p, label, D1, D2, D3])
                    f.write(f"{L},{p},{label},{D1:.6e},{D2:.6e},{D3:.6e}\n")
                    f.flush()
                    print(f"  {label:<15s}  ||w1-tgt||={D1:.4e}  "
                          f"||cmb-tgt||={D2:.4e}  ||w1-cmb||={D3:.4e}", flush=True)
    print(f"\nSaved {csv}")


if __name__ == "__main__":
    main()
