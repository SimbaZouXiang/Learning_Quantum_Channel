"""Truncation-error spot check at N=30 for the random-unitary experiments.

The paper truncates the N=30 sweeps at d_max=32 but quantifies truncation error
only at N=12.  Here we push a few weight-1 Pauli inputs through the SAME
teacher (fixed seed) at d_max in {32, 64, 128} and report the normalized
Frobenius distance between the d_max outputs, at the worst-case corners of the
grid (small p, large L).

Forward passes only — no training.  Runs comfortably inside the packed job.
"""
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TRUNC_THREADS", "32")))

from qcl.models import QMLM_output_only  # noqa: E402
from qcl.states import Identity_init, operator_assignment_single_site  # noqa: E402


def make_input(N, site, op):
    M = Identity_init(N, bond_dim=1, phys_dim=4)
    operator_assignment_single_site(M, site, op)
    M.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    return M


def frob_dist(A, B):
    a = A.copy(); b = B.copy()
    xAA = complex(a.H @ a); xBB = complex(b.H @ b); xAB = complex(a.H @ b)
    d2 = abs(xAA) + abs(xBB) - 2 * xAB.real
    return float(max(d2, 0.0) ** 0.5), float(abs(xAA) ** 0.5), float(abs(xBB) ** 0.5)


def main():
    N = 30
    cases = [(0.04, 5), (0.04, 7), (0.08, 7)]      # (p, L): small-p / deep = worst case
    bonds = [32, 64, 128]
    inputs = [(N // 2 - 1, "X"), (N // 2 - 1, "Z"), (2, "X")]
    out = []
    for (p, L) in cases:
        torch.manual_seed(7)
        np.random.seed(7)
        p_dep = torch.ones((L, N), dtype=torch.float64) * p
        teacher = QMLM_output_only(N, L, p_depolar=p_dep, max_bd=max(bonds))
        for (site, op) in inputs:
            outs = {}
            for bd in bonds:
                teacher.max_bd = bd
                M = make_input(N, site, op)
                with torch.no_grad():
                    Mo = teacher.forward_depolarizing_only(M, truncation=True)
                outs[bd] = Mo
            for bd in bonds[:-1]:
                d, na, nb = frob_dist(outs[bd], outs[bonds[-1]])
                rel = d / max(nb, 1e-30)
                rec = dict(p=p, L=L, site=site, op=op, bd=bd, ref_bd=bonds[-1],
                           dist=d, norm_ref=nb, rel_dist=rel)
                out.append(rec)
                print(f"p={p} L={L} {op}@{site}: d_max={bd} vs {bonds[-1]}: "
                      f"dist={d:.3e} rel={rel:.3e}", flush=True)
    path = os.path.join(SCRIPT_DIR, "loe_measurement", "truncation_check_N30.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"))
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
