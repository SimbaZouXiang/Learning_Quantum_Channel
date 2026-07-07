"""Plot 4 training variants vs L_target on three eval sets and two noise levels.

  w1         (24 structured inputs)         — from mpo_save_params/
  combined   (276 structured w1+w2-full)    — from mpo_save_params/
  random24   (24 random Pauli inputs)       — from mpo_random_test/
  random276  (276 random Pauli inputs)      — from mpo_random_test/

2x3 grid:  rows = p ∈ {0.01, 0.05},   cols = eval set ∈ {weight1, weight2, random}.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['figure.dpi'] = 200

ROOT = os.path.dirname(__file__)
RD_RND = os.path.join(ROOT, "results")
RD_SP  = os.path.join(os.path.dirname(ROOT), "mpo_save_params", "results")
FD     = os.path.join(ROOT, "figures")
os.makedirs(FD, exist_ok=True)

N, T = 8, 3
DEPOL_LIST = [0.01, 0.05]
LTARGETS = [3, 4, 5, 6, 7]


def _load_mean(rd, tag, eval_kind):
    p = os.path.join(rd, f"{tag}_eval_{eval_kind}.npy")
    if not os.path.exists(p):
        return float('nan')
    return float(np.load(p, allow_pickle=True).mean())


def main():
    eval_sets = [("weight1", "weight-1 inputs (24)"),
                  ("weight2", "weight-2 inputs (84)"),
                  ("random",  "random Paulis (30)")]
    fig, axes = plt.subplots(len(DEPOL_LIST), len(eval_sets),
                              figsize=(5.0 * len(eval_sets), 4.0 * len(DEPOL_LIST)),
                              sharex=True, squeeze=False)
    for pi, p in enumerate(DEPOL_LIST):
        for ei, (es, title) in enumerate(eval_sets):
            ax = axes[pi, ei]
            series = [
                ("w1 (24, structured)",        RD_SP,  lambda L,p=p: f"N{N}_T{T}_L{L}_p{p}_w1",        "C0", "o", "-"),
                ("combined (276, structured)", RD_SP,  lambda L,p=p: f"N{N}_T{T}_L{L}_p{p}_combined",  "C3", "s", "-"),
                ("random 24",                  RD_RND, lambda L,p=p: f"N{N}_T{T}_L{L}_p{p}_random24",  "C0", "^", "--"),
                ("random 276",                 RD_RND, lambda L,p=p: f"N{N}_T{T}_L{L}_p{p}_random276", "C3", "D", "--"),
            ]
            for label, rd, tag_fn, color, marker, ls in series:
                means = [_load_mean(rd, tag_fn(L), es) for L in LTARGETS]
                ax.plot(LTARGETS, means, marker=marker, color=color,
                        linestyle=ls, label=label)
            ax.set_xlabel(r"$L_{\rm target}$")
            ax.set_ylabel("mean eval loss")
            ax.set_title(rf"$p={p}$,  {title}")
            ax.grid(True, alpha=0.25)
            ax.set_xticks(LTARGETS)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)

    fig.suptitle(
        rf"Structured vs random Pauli training inputs at matched sample-count "
        rf"($N={N}$, $T_{{\rm model}}={T}$, MPO target, 200 + 40 epochs)",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("png", "pdf"):
        out = os.path.join(FD, f"four_variant_compare.{ext}")
        fig.savefig(out)
        print(f"Saved {out}")

    # Summary CSV
    csv = os.path.join(FD, "four_variant_compare_summary.csv")
    with open(csv, "w") as f:
        f.write("L,p,eval_set,w1_24,combined_276,random_24,random_276\n")
        for p in DEPOL_LIST:
            for L in LTARGETS:
                for es, _ in eval_sets:
                    w1   = _load_mean(RD_SP, f"N{N}_T{T}_L{L}_p{p}_w1",        es)
                    cmb  = _load_mean(RD_SP, f"N{N}_T{T}_L{L}_p{p}_combined",  es)
                    r24  = _load_mean(RD_RND, f"N{N}_T{T}_L{L}_p{p}_random24",  es)
                    r276 = _load_mean(RD_RND, f"N{N}_T{T}_L{L}_p{p}_random276", es)
                    f.write(f"{L},{p},{es},{w1:.4e},{cmb:.4e},{r24:.4e},{r276:.4e}\n")
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
