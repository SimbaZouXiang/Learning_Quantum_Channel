"""Compare random-input training vs structured-Pauli training.

Compares the four training variants (load from prior runs + this run):
  - w1            (24 weight-1 inputs)         in ../mpo_save_params/
  - random24      (24 random Pauli inputs)     in ./results/
  - combined      (276 = w1 + w2-full)         in ../mpo_save_params/
  - random276     (276 random Pauli inputs)    in ./results/

Reports per (L, p, eval_set) the mean per-sample tn_distance (= Frobenius
NORM of output difference, as used everywhere in this project) on three
eval sets: weight-1 (24), weight-2 same-op (84), random Paulis (30).
"""
import os
import numpy as np

ROOT = os.path.dirname(__file__)
RD_RANDOM = os.path.join(ROOT, "results")
RD_PRIOR  = os.path.join(os.path.dirname(ROOT), "mpo_save_params", "results")
FD = os.path.join(ROOT, "figures")
os.makedirs(FD, exist_ok=True)

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS = [3, 4, 5, 6, 7]
EVAL_SETS = ["weight1", "weight2", "random"]

VARIANTS = [
    ("w1",        24,  RD_PRIOR),
    ("random24",  24,  RD_RANDOM),
    ("combined",  276, RD_PRIOR),
    ("random276", 276, RD_RANDOM),
]


def _load(rd, tag, eval_set):
    p = os.path.join(rd, f"{tag}_eval_{eval_set}.npy")
    return np.load(p, allow_pickle=True).astype(float) if os.path.exists(p) else None


def main():
    csv = os.path.join(FD, "random_vs_pauli_summary.csv")
    with open(csv, "w") as f:
        f.write("L,p,eval_set,w1_mean,random24_mean,combined_mean,random276_mean\n")
        for L in LTARGETS:
            for p in DEPOL_LIST:
                for es in EVAL_SETS:
                    row = [L, p, es]
                    for vt, _, rd in VARIANTS:
                        tag = f"N{N}_T{T}_L{L}_p{p}_{vt}"
                        a = _load(rd, tag, es)
                        row.append(a.mean() if a is not None else float('nan'))
                    f.write(",".join([str(row[0]), str(row[1]), row[2]]
                                     + [f"{x:.4e}" for x in row[3:]]) + "\n")
    print(f"Saved {csv}")

    # Print headline table
    print()
    print(f"{'L':>3} {'p':>5} {'eval':>10}  "
          f"{'w1 (24)':>10}  {'rand24':>10}  {'combined':>10}  {'rand276':>10}")
    for L in LTARGETS:
        for p in DEPOL_LIST:
            for es in EVAL_SETS:
                vals = []
                for vt, _, rd in VARIANTS:
                    tag = f"N{N}_T{T}_L{L}_p{p}_{vt}"
                    a = _load(rd, tag, es)
                    vals.append(a.mean() if a is not None else float('nan'))
                print(f"{L:>3} {p:>5} {es:>10}  "
                      f"{vals[0]:>10.3e}  {vals[1]:>10.3e}  "
                      f"{vals[2]:>10.3e}  {vals[3]:>10.3e}")
            print()


if __name__ == "__main__":
    main()
