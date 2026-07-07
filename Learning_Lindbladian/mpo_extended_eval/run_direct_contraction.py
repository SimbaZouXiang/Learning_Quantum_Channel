"""Direct MPO-MPO Frobenius² distance for all 5 students + target per cell.

For each (L, p) cell:
  Load the six channels (w1, w2full, combined, random24, random276, target)
  as QMLM modules. Build each as an MPO via model.get_MPO(noise_type='depolarizing').
  Compute the C(6,2)=15 pairwise squared Frobenius distances using direct
  tensor-network contraction (Tr(A^† A), Tr(B^† B), Tr(A^† B)).

  ||A - B||_F^2 = Tr(A^† A) + Tr(B^† B) - 2 Re Tr(A^† B)

This gives the EXACT channel Frobenius distance (the squared one, not the
norm); no input sampling, no per-sample bias.

CSV columns: L, p, A, B, AA, BB, ReAB, frob_sq
"""
import os, sys, time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

torch.set_num_threads(16)
import itertools
import quimb.tensor as qtn
import TDME_Trott as tdme

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS   = [3, 4, 5, 6, 7]
MAX_BD, MAX_ERR = 32, 1e-6
NOISE_TYPE = "depolarizing"

# Student variants (tag, folder).
STUDENTS = [
    ("w1",        "mpo_save_params"),
    ("w2full",    "mpo_save_params"),
    ("combined",  "mpo_save_params"),
    ("random24",  "mpo_random_test"),
    ("random276", "mpo_random_test"),
]
CHANNELS = [s[0] for s in STUDENTS] + ["target"]

OUT_CSV = os.path.join(SCRIPT_DIR, "results", "direct_contraction_distances.csv")


def _load_student_qmlm(L, p, tag, folder):
    prefix = os.path.join(PARENT_DIR, folder, "results",
                          f"N{N}_T{T}_L{L}_p{p}_{tag}")
    params_np    = np.load(prefix + "_params.npy",   allow_pickle=True)
    p_depolar    = np.load(prefix + "_p_depolar.npy", allow_pickle=True)
    p_dephaseX   = np.load(prefix + "_p_dephaseX.npy", allow_pickle=True)
    p_dephaseY   = np.load(prefix + "_p_dephaseY.npy", allow_pickle=True)
    p_dephaseZ   = np.load(prefix + "_p_dephaseZ.npy", allow_pickle=True)
    model = tdme.QMLM(N, T, max_bd=MAX_BD, max_err=MAX_ERR)
    model.params = torch.nn.Parameter(
        torch.from_numpy(params_np).to(torch.complex128), requires_grad=False,
    )
    model.p_depolar.data  = torch.from_numpy(p_depolar).to(torch.float64)
    model.p_dephaseX.data = torch.from_numpy(p_dephaseX).to(torch.float64)
    model.p_dephaseY.data = torch.from_numpy(p_dephaseY).to(torch.float64)
    model.p_dephaseZ.data = torch.from_numpy(p_dephaseZ).to(torch.float64)
    return model


def _load_target_qmlm(L, p):
    """Build QMLM for the target channel from saved teacher params."""
    target_param = torch.load(
        os.path.join(PARENT_DIR, "mpo_save_params", "results",
                     f"N{N}_T{T}_L{L}_p{p}_w1_target_param.pt"),
        map_location="cpu", weights_only=True,
    )
    p_depolar_tgt = torch.ones((L, N), dtype=torch.float64) * p
    # Construct QMLM with L layers (matching target_param shape). The QMLM
    # forward and get_MPO handle the depth from self.layers.
    model = tdme.QMLM(N, L, max_bd=MAX_BD, max_err=MAX_ERR)
    model.params = torch.nn.Parameter(
        target_param.to(torch.complex128), requires_grad=False,
    )
    model.p_depolar.data  = p_depolar_tgt
    model.p_dephaseX.data = torch.zeros((L, N), dtype=torch.float64)
    model.p_dephaseY.data = torch.zeros((L, N), dtype=torch.float64)
    model.p_dephaseZ.data = torch.zeros((L, N), dtype=torch.float64)
    return model


def _to_plain_complex128(tn):
    """Unparametrize all PTensors and cast every tensor data to complex128."""
    new_tensors = []
    for t in tn.tensors:
        plain = t.unparametrize() if hasattr(t, "unparametrize") else t.copy()
        if torch.is_tensor(plain.data):
            plain.modify(data=plain.data.to(torch.complex128))
        else:
            plain.modify(data=plain.data.astype(np.complex128))
        new_tensors.append(plain)
    return qtn.TensorNetwork(new_tensors)


def _rename_internal(tn, suffix):
    physical = set(f'input{i}' for i in range(N)) | set(f'k{i}' for i in range(N))
    rename_map = {}
    for t in tn.tensors:
        for ind in t.inds:
            if ind not in physical and ind not in rename_map:
                rename_map[ind] = f"{ind}_{suffix}"
    return tn.reindex(rename_map)


def hs_inner(A_plain, B_plain, suffix_a="A", suffix_b="B"):
    """Compute Tr(A^† B) = sum_{β,α} A[β,α]* B[β,α] via contraction."""
    # Conjugate A entrywise.
    A_conj = A_plain.copy()
    for t in A_conj.tensors:
        t.modify(data=t.data.conj() if torch.is_tensor(t.data) else np.conj(t.data))
    A_renamed = _rename_internal(A_conj, suffix_a)
    B_renamed = _rename_internal(B_plain.copy(), suffix_b)
    combined = A_renamed & B_renamed
    return complex(combined.contract().item())


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    # Resume support: read existing CSV and skip (L, p) cells already complete
    done_cells = set()
    if os.path.exists(OUT_CSV):
        counts = {}
        with open(OUT_CSV) as f:
            next(f, None)  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    key = (int(parts[0]), float(parts[1]))
                    counts[key] = counts.get(key, 0) + 1
        # 15 pairs per cell → done
        done_cells = {k for k, v in counts.items() if v >= 15}
        print(f"  Resume: {len(done_cells)} cells already complete", flush=True)
        mode = "a"
    else:
        mode = "w"

    with open(OUT_CSV, mode) as f:
        if mode == "w":
            f.write("L,p,A,B,AA,BB,ReAB,frob_sq\n")
            f.flush()
        for L in LTARGETS:
            for p in DEPOL_LIST:
                if (L, p) in done_cells:
                    print(f"  SKIP L={L} p={p}  (already complete)", flush=True)
                    continue
                t0 = time.time()
                # Load all 6 channels for this cell.
                models = {}
                for tag, folder in STUDENTS:
                    models[tag] = _load_student_qmlm(L, p, tag, folder)
                models["target"] = _load_target_qmlm(L, p)
                # Build MPOs and convert to plain TNs once.
                mpos = {}
                for tag in CHANNELS:
                    raw_mpo = models[tag].get_MPO(noise_type=NOISE_TYPE)
                    mpos[tag] = _to_plain_complex128(raw_mpo)
                # Precompute self-overlaps.
                self_overlaps = {}
                for tag in CHANNELS:
                    AA = hs_inner(mpos[tag], mpos[tag], suffix_a=f"{tag}A", suffix_b=f"{tag}B")
                    self_overlaps[tag] = AA
                # All C(6, 2) = 15 pairs.
                pairs = list(itertools.combinations(CHANNELS, 2))
                for (a, b) in pairs:
                    AB = hs_inner(mpos[a], mpos[b], suffix_a=f"{a}", suffix_b=f"{b}")
                    AA = self_overlaps[a]
                    BB = self_overlaps[b]
                    frob_sq = float(np.real(AA + BB - 2 * AB))
                    f.write(f"{L},{p},{a},{b},{np.real(AA):.6e},{np.real(BB):.6e},"
                            f"{np.real(AB):.6e},{frob_sq:.6e}\n")
                    f.flush()
                elapsed = time.time() - t0
                print(f"  L={L} p={p}  done in {elapsed:.1f}s", flush=True)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
