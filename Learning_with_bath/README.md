# Learning with bath

Experiments and drivers for the **QMLM-with-bath** learning setup (teacher = 2N-qubit
`d b d b …` layout with Haar data-data + weak data-bath gates, student = plain
N-qubit QMLM with trainable noise). All the bath-side code lives in
[`../Learning_Lindbladian/TDME_Trott.py`](../Learning_Lindbladian/TDME_Trott.py):
`QMLM_with_bath_output_only`, `Learning_QMLM_with_bath_scheduler`,
`partial_trace_bath`, etc.

## Layout

```
Learning_with_bath/
├── run_bath_N6.py          # single training run (N=6, T=3, L=6 by default)
├── run_bath_N6_sweep.py    # one g per SLURM array index (0..5 → 0.05..0.30)
├── audit_N6_consistency.py # g=0 + matched-Haar sanity (element-wise)
├── plot_bath_N6_sweep.py   # loss-curve overlay + final-loss-vs-g plot
├── job_bath_N6.sh          # sbatch wrapper for run_bath_N6.py
├── job_bath_N6_sweep.sh    # sbatch array wrapper (array=0-5)
├── job_audit_N6.sh         # sbatch wrapper for audit
├── run_bath_N6_in_debugjob.sh # interactive-compute wrapper (rarely useful; login CPU rlimit inherits)
├── Figures/                # .png outputs
├── npy_outputs/            # .npy learning/test loss + trained params
└── slurm_outputs/          # SLURM stdout captures + debugjob logs
```

## Running

All SLURM scripts `cd` into this directory, so submit from anywhere:

```bash
# single run
sbatch Learning_with_bath/job_bath_N6.sh

# coupling-strength sweep (g = 0.05, 0.10, ..., 0.30)
sbatch Learning_with_bath/job_bath_N6_sweep.sh

# consistency audit (finishes in ~1 min)
sbatch Learning_with_bath/job_audit_N6.sh
```

After the sweep completes:

```bash
cd Learning_with_bath
python plot_bath_N6_sweep.py
```

## Notes / known caveats

- `use_compressed=True` in the student QMLM hits a pre-existing canonicalization
  bug for N≥4 with T≥2 (reproducible outside the bath code). All driver scripts
  set `use_compressed=False` — slower but stable. If you need speed at larger
  N, fix the upstream bug first.
- Per-sub-layer compression in `QMLM_with_bath_output_only.forward` keeps bond
  dims bounded at `max_bd` (default 16 here); without it, L≥3 blows up.
- `NUMBA_CACHE_DIR` is set to `$SLURM_TMPDIR` in the sbatch scripts because
  the login home directory is read-only on compute nodes.

## Sweep results so far (N=6, T=3, L=6, 100 epochs, depolarizing student)

| g    | final train loss | avg test loss |
|------|------------------|---------------|
| 0.05 | 0.535            | 0.439         |
| 0.10 | 0.352            | 0.263         |
| 0.15 | 0.186            | 0.117         |
| 0.20 | 0.087            | 0.041         |
| 0.25 | 0.044            | 0.016         |
| 0.30 | 0.028            | 0.009         |

Both train and test loss drop monotonically as the system-bath coupling
strength increases; with stronger coupling the effective data-side channel
becomes more "noise-like" and easier for the depolarizing student to match.

## Audit status

- Dense Pauli-basis reference (N=2, L=2, g=0.1): machine-precision agreement.
- Matched-gate g=0 consistency (N=6, L=T=3, 18 inputs): MAX element-wise
  diff 4.16e-16.
