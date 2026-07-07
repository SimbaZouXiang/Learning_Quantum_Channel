"""Pairwise distance analysis with three trained students: w1, w2-full, combined.

For each (L, p) cell:
  1. Reconstruct each student model from saved .npy params.
  2. Build the target QMLM module.
  3. For each of three input sets (weight-1, weight-2-same-op, random):
       For each input P:
         out_w1  = M_w1(P),  out_w2  = M_w2full(P),  out_cmb = M_combined(P),  tgt = target(P)
         compute 6 pairwise distances:
            D(w1, target), D(w2, target), D(cmb, target),
            D(w1, w2),     D(w1, cmb),    D(w2, cmb)
       average over inputs.

Distances here use tensor_network_distance, which returns the Frobenius
NORM (not the squared norm — see line 140 of quimb.tensor.fitting). So the
reported numbers are mean per-sample Frobenius distances of output operators
— same metric used in training/testing throughout this project.

CSV columns:
  L, p, eval_set, D_w1_tgt, D_w2_tgt, D_cmb_tgt, D_w1_w2, D_w1_cmb, D_w2_cmb
"""
import os, sys, warnings
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch
torch.set_num_threads(16)
import TDME_Trott as tdme

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS = [3, 4, 5, 6, 7]
MAX_BD, MAX_ERR = 32, 1e-6
NOISE_TYPE = "depolarizing"
TRUNCATION = False
N_RANDOM = 500

RD = os.path.join(SCRIPT_DIR, "results")
FD = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FD, exist_ok=True)


def _load_student(L, p, variant_tag):
    """Load trained QMLM(N, T) from saved .npy params (bypasses state_dict issue)."""
    prefix = os.path.join(RD, f"N{N}_T{T}_L{L}_p{p}_{variant_tag}")
    params_np    = np.load(prefix + "_params.npy",   allow_pickle=True)
    p_depolar    = np.load(prefix + "_p_depolar.npy", allow_pickle=True)
    p_dephaseX   = np.load(prefix + "_p_dephaseX.npy", allow_pickle=True)
    p_dephaseY   = np.load(prefix + "_p_dephaseY.npy", allow_pickle=True)
    p_dephaseZ   = np.load(prefix + "_p_dephaseZ.npy", allow_pickle=True)
    model = tdme.QMLM(N, T, max_bd=MAX_BD, max_err=MAX_ERR)
    model.params = torch.nn.Parameter(
        torch.from_numpy(params_np).to(torch.complex128), requires_grad=False,
    )
    model.p_depolar.data    = torch.from_numpy(p_depolar).to(torch.float64)
    model.p_dephaseX.data   = torch.from_numpy(p_dephaseX).to(torch.float64)
    model.p_dephaseY.data   = torch.from_numpy(p_dephaseY).to(torch.float64)
    model.p_dephaseZ.data   = torch.from_numpy(p_dephaseZ).to(torch.float64)
    return model


def _build_target_module(L, p):
    target_param = torch.load(
        os.path.join(RD, f"N{N}_T{T}_L{L}_p{p}_w1_target_param.pt"),
        map_location="cpu", weights_only=True,
    )
    p_depolar_tgt = torch.ones((L, N), dtype=torch.float64) * p
    return tdme.QMLM_output_only(
        N, L, param=target_param, p_depolar=p_depolar_tgt,
        max_bd=MAX_BD, max_err=MAX_ERR,
    )


def _compute_pairwise(students, target_module, input_mps_list):
    """Compute six pairwise mean distances over the input set.

    students: dict {'w1': model_w1, 'w2': model_w2full, 'cmb': model_combined}
    Returns dict of six mean tn_distance values:
      'w1_tgt', 'w2_tgt', 'cmb_tgt', 'w1_w2', 'w1_cmb', 'w2_cmb'
    """
    sums = {k: 0.0 for k in ['w1_tgt','w2_tgt','cmb_tgt','w1_w2','w1_cmb','w2_cmb']}
    count = 0
    with torch.no_grad():
        mpo_w1  = students['w1'].get_MPO(noise_type=NOISE_TYPE)
        mpo_w2  = students['w2'].get_MPO(noise_type=NOISE_TYPE)
        mpo_cmb = students['cmb'].get_MPO(noise_type=NOISE_TYPE)
        for inp in input_mps_list:
            tgt = target_module.forward_depolarizing_only(inp.copy(), truncation=TRUNCATION)
            for i, ten in enumerate(tgt):
                ten.reindex_({f'input{i}': f'k{i}'})
            tgt_c = tgt.astype("complex128")
            out_w1  = (mpo_w1  | inp).astype("complex128")
            out_w2  = (mpo_w2  | inp).astype("complex128")
            out_cmb = (mpo_cmb | inp).astype("complex128")
            sums['w1_tgt'] += tdme.tensor_network_distance(out_w1,  tgt_c).item()
            sums['w2_tgt'] += tdme.tensor_network_distance(out_w2,  tgt_c).item()
            sums['cmb_tgt']+= tdme.tensor_network_distance(out_cmb, tgt_c).item()
            sums['w1_w2']  += tdme.tensor_network_distance(out_w1,  out_w2).item()
            sums['w1_cmb'] += tdme.tensor_network_distance(out_w1,  out_cmb).item()
            sums['w2_cmb'] += tdme.tensor_network_distance(out_w2,  out_cmb).item()
            count += 1
    return {k: v / count for k, v in sums.items()}


def main():
    inputs_w1 = tdme.Pauli_MPS_weight_1(N)
    inputs_w2 = tdme.Pauli_MPS_weight_2(N)
    np.random.seed(1234); torch.manual_seed(1234)
    inputs_rand = [tdme.random_pauli_MPS(N)[0] for _ in range(N_RANDOM)]
    eval_sets = [("weight1_full", inputs_w1),
                  ("weight2_full", inputs_w2),
                  (f"random_{N_RANDOM}", inputs_rand)]

    csv = os.path.join(FD, "three_student_pairwise_distances.csv")
    # Resume support: read which (L, p, eval_set) cells are already done.
    done = set()
    if os.path.exists(csv):
        with open(csv) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    done.add((int(parts[0]), float(parts[1]), parts[2]))
        print(f"  Resuming: {len(done)} cells already on disk", flush=True)
        mode = "a"
    else:
        mode = "w"

    with open(csv, mode) as f:
        if mode == "w":
            f.write("L,p,eval_set,D_w1_tgt,D_w2_tgt,D_cmb_tgt,D_w1_w2,D_w1_cmb,D_w2_cmb\n")
            f.flush()
        for L in LTARGETS:
            for p in DEPOL_LIST:
                # Skip if all three eval sets for this cell are already done.
                already_done_cell = all(
                    (L, p, label) in done for label, _ in eval_sets
                )
                if already_done_cell:
                    print(f"  SKIP  L={L} p={p}  (all 3 eval sets already done)", flush=True)
                    continue
                students = {
                    'w1':  _load_student(L, p, 'w1'),
                    'w2':  _load_student(L, p, 'w2full'),
                    'cmb': _load_student(L, p, 'combined'),
                }
                target_module = _build_target_module(L, p)
                print(f"\n=== L={L} p={p} ===", flush=True)
                for label, inputs in eval_sets:
                    if (L, p, label) in done:
                        print(f"  SKIP {label} (already on disk)", flush=True)
                        continue
                    d = _compute_pairwise(students, target_module, inputs)
                    f.write(f"{L},{p},{label},"
                            f"{d['w1_tgt']:.6e},{d['w2_tgt']:.6e},{d['cmb_tgt']:.6e},"
                            f"{d['w1_w2']:.6e},{d['w1_cmb']:.6e},{d['w2_cmb']:.6e}\n")
                    f.flush()
                    print(f"  {label:<15s}  "
                          f"w1-tgt={d['w1_tgt']:.3e}  w2-tgt={d['w2_tgt']:.3e}  "
                          f"cmb-tgt={d['cmb_tgt']:.3e}", flush=True)
                    print(f"  {'':<15s}  "
                          f"w1-w2 ={d['w1_w2']:.3e}  w1-cmb={d['w1_cmb']:.3e}  "
                          f"w2-cmb={d['w2_cmb']:.3e}", flush=True)
    print(f"\nSaved {csv}")


if __name__ == "__main__":
    main()
