"""Measure the max bipartite Local Operator Entanglement (LOE) of the *target*
channel output as a function of circuit depth L, for several per-layer
depolarizing strengths p.

Validates the Sec.-VII ansatz a(L) ~ v_E * L * q^(L^2+2L) of the paper
(Quantum_Channel_Learning.tex): measured peak location vs the corrected
stationary point L* = (-1 + sqrt(1 - 2/ln q)) / 2.

Layer convention == qcl teacher (odd brick, even brick, per-site depolarizing),
i.e. one layer of Pauli_MPS_after_QMLM_output_only with noise_type="depolarizing".
Entropies are computed EXACTLY by densifying the vectorized-operator MPS
(4^N amplitudes; N<=12 with truncation during evolution only).

Usage:
    python measure_loe.py --N 8  --Lmax 10 --seeds 2 --out loe_measurement
    python measure_loe.py --N 12 --Lmax 12 --seeds 2 --max-bd 256 --truncation --out loe_measurement
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("LOE_THREADS", "4")))

from qcl.pauli import construct_all_SU4  # noqa: E402
from qcl.noise import depolarization_noise_transfer_matrix  # noqa: E402
from qcl.evolve import two_site_layer, one_site_layer  # noqa: E402
from qcl.states import Identity_init, operator_assignment_single_site  # noqa: E402


def dense_vector(M):
    """Contract the (operator-vectorized) MPS to a dense 4^N vector (numpy)."""
    Mc = M.copy()
    Mc.apply_to_arrays(lambda x: np.asarray(torch.as_tensor(x).detach(), dtype=np.complex128))
    t = Mc.contract(..., optimize="auto-hq")
    t = t.squeeze()
    order = [f"input{i}" for i in range(M.L)]
    t.transpose_(*order)
    return np.asarray(t.data).reshape(-1)


def entropies_from_vector(vec, N, phys=4):
    """Max-over-bonds von Neumann entropy (nats) + Renyi-0/inf at that bond."""
    nrm = np.linalg.norm(vec)
    if nrm == 0:
        return dict(maxS1=0.0, argbond=0, S0=0.0, Sinf=0.0, profile=[0.0] * (N - 1))
    v = vec / nrm
    best = (-1.0, 0, 0.0, 0.0)
    profile = []
    for k in range(1, N):
        A = v.reshape(phys ** k, phys ** (N - k))
        s = np.linalg.svd(A, compute_uv=False)
        p = s ** 2
        p = p / p.sum()
        p_pos = p[p > 1e-14]
        S1 = float(-(p_pos * np.log(p_pos)).sum())
        profile.append(S1)
        if S1 > best[0]:
            S0 = float(np.log(len(p[p > 1e-12])))
            Sinf = float(-np.log(p_pos.max()))
            best = (S1, k, S0, Sinf)
    return dict(maxS1=best[0], argbond=best[1], S0=best[2], Sinf=best[3], profile=profile)


def make_input(N, site, op):
    M = Identity_init(N, bond_dim=1, phys_dim=4)
    operator_assignment_single_site(M, site, op)
    M.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    return M


def run_pass(N, Lmax, p, seed, site, op, truncation, max_bd):
    """One incremental evolution; returns per-layer entropy records L=1..Lmax."""
    torch.manual_seed(seed)
    param = (torch.rand(Lmax, N, 16, dtype=torch.float64)
             + 1j * torch.rand(Lmax, N, 16, dtype=torch.float64))
    all_U = construct_all_SU4(N, Lmax, param)
    noise = [depolarization_noise_transfer_matrix(torch.tensor(p, dtype=torch.float64))
             for _ in range(N)]
    M = make_input(N, site, op)
    records = []
    with torch.no_grad():
        for layer in range(Lmax):
            U_list = all_U[layer, :N]
            _, M = two_site_layer(M, U_list[: N // 2], "o",
                                  truncation=truncation, max_bd=max_bd, max_err=1e-12)
            M.right_canonicalize()
            _, M = two_site_layer(M, U_list[N // 2:], "e",
                                  truncation=truncation, max_bd=max_bd, max_err=1e-12)
            _, M = one_site_layer(M, noise, max_bd=max_bd, max_err=1e-12)
            if truncation:
                M.right_canonicalize()
                M.compress(form="right", max_bond=max_bd, cutoff=1e-12)
            rec = entropies_from_vector(dense_vector(M), N)
            rec.update(L=layer + 1, p=p, seed=seed, site=site, op=op)
            records.append(rec)
            print(f"  p={p:<6} seed={seed} {op}@{site}: L={layer+1:2d} "
                  f"maxS1={rec['maxS1']:.4f} (bond {rec['argbond']}, "
                  f"S0={rec['S0']:.2f}, Sinf={rec['Sinf']:.2f})", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--Lmax", type=int, default=10)
    ap.add_argument("--plist", type=str, default="0,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--truncation", action="store_true")
    ap.add_argument("--max-bd", type=int, default=4096)
    ap.add_argument("--out", type=str, default="loe_measurement")
    args = ap.parse_args()

    N = args.N
    plist = [float(x) for x in args.plist.split(",")]
    inputs = [(N // 2 - 1, "X"), (N // 2 - 1, "Z"), (1, "X")]
    os.makedirs(args.out, exist_ok=True)

    all_records = []
    for p in plist:
        for seed in range(args.seeds):
            for site, op in inputs:
                all_records += run_pass(N, args.Lmax, p, seed, site, op,
                                        args.truncation, args.max_bd)

    tag = f"N{N}_Lmax{args.Lmax}_seeds{args.seeds}" + ("_trunc%d" % args.max_bd if args.truncation else "_exact")
    path = os.path.join(args.out, f"loe_{tag}.json")
    with open(path, "w") as f:
        json.dump(all_records, f)
    print("WROTE", path, f"({len(all_records)} records)", flush=True)


if __name__ == "__main__":
    main()
