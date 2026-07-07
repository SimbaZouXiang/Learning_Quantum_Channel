# Quantum Circuit Learning

Research code for **learning noisy quantum channels** (Lindbladian dynamics)
with tensor-network parameterised variational models, in the vectorised Pauli
basis. Runs on a Compute-Canada / SLURM cluster (`~/.virtualenvs/QIP`).

## Layout

```
qcl/                      ← THE library (was Learning_Lindbladian/TDME_Trott.py)
  backend.py                cotengra path cache, tensor_network_distance wrapper,
                            global torch.linalg.svd safety patch (applied on import)
  pauli.py                  Pauli-basis constants, unitary→transfer-matrix, SU(4)
  noise.py                  depolarizing / dephasing transfer matrices
  states.py                 identity/product MPS + Pauli-string input bases
  mpo.py                    MPO-building forward path (exact, exponential in depth)
  evolve.py                 layer-by-layer compressed MPS path (O(N·D³) per layer)
  bath.py                   system-bath (2N-site) teacher + partial trace + data gen
  models.py                 QMLM (student), QMLM_output_only (teacher)
  tdme.py                   Trotterised master-equation physics + TDME model
  io.py                     save_mps / load_mps (.npz MPS serialisation)
  datagen.py                (input, target) MPS pair generation (+ pool workers)
  training.py               Learning_* / Testing_* loops

Learning_Lindbladian/     ← TDME/Lindbladian experiments
  TDME_Trott.py             ★ compat shim only — re-exports qcl; do not add code here
  driver_shard.py           parameterized (t, γ) shard driver  + job_t*_*.sh wrappers
  run_trotterization*_one.py  SLURM-array Trotter-error drivers + job_*_array.sh
  generate_TDME_data.py     dataset generator (argparse)
  plot_*.py                 figure scripts (read Learning_result/, write Figures/)
  Learning_data/            cached (input, target) MPS datasets  (do not move)
  Learning_result/          trained params / losses               (do not move)
  mpo_*/ tdme_save_params/ weight1_vs_weight2*/   self-contained experiments
  legacy_drivers/           superseded clone families (Learning_TDME_using_data_*,
                            Trotterization_3_to_30_*, generate_TDME_data{24,68,1020})
                            — replaced by driver_shard.py / run_trotterization_one.py /
                            generate_TDME_data.py. Kept for reference; their
                            job scripts assume the parent dir as CWD.
  scratch/                  one-off tests & debug scripts (not maintained)

Learning_with_bath/       ← system-bath experiments (imports the shim via sys.path)
Learning_unitary/           load_training_data.py = shim → qcl.io; data generator
run_depolarizing_sweep.py ← replaces the 14 generated LRU_p*_L*.py clones
job_depolarizing_sweep.sh   sbatch wrapper: forwards its args to the driver
Learning_random_unitary.py  N=8 random-unitary variant (+ job.sh)
legacy_generated/           the old LRU_*.py / job_p*_L*.sh stamps + generate_files.py
scratch/                    top-level one-off benchmarks / micro-tests
```

## Using the library

New code:

```python
from qcl import QMLM, Learning_MPO_scheduler, tensor_network_distance
```

Old drivers keep working unchanged — `import TDME_Trott as tdme` resolves to
the shim, which puts the project root on `sys.path` and re-exports every name
(including private helpers) from `qcl`. Optionally install it properly:

```bash
pip install -e "/scratch/simba/Quantum Circuit Learning"
```

## Invariants (see CLAUDE.md for detail)

- `qcl.backend` must be imported before any quimb compression runs; importing
  `qcl` (or the shim) does this automatically. Do not construct fresh cotengra
  optimisers — always go through `qcl.backend.tensor_network_distance`.
- All trainable unitaries flow through `construct_SU4_from_input`.
- Physical-leg naming: inputs `input{i}`, targets/outputs `k{i}`; reindex
  `input{i} → k{i}` before every `tensor_network_distance` call.
- Data directories (`Learning_data/`, `Learning_result/`, `npy_outputs/`,
  per-experiment `results/`) are referenced script-relative or CWD-relative:
  don't move them, and launch jobs from the directory that owns them.
- `/scratch/simba/Learning_Lindbladian` and `/scratch/simba/Learning_with_bath`
  are symlinks used by the SLURM scripts — directory names must not change.

## Sweeps

```bash
# was: sbatch job_p0_L345.sh  (LRU_p0_L345.py)
sbatch job_depolarizing_sweep.sh --p-list 0 --L-list 3 4 5
# was: sbatch job_p6_L67.sh   (8 threads/worker to avoid OOM at L>=6)
sbatch job_depolarizing_sweep.sh --p-list 6 --L-list 6 7 --threads-per-worker 8
```

Outputs keep the historic `Depolarizing_N{N}_T{T}_L{L}_p_{p:03d}_*.npy` naming
(default in the CWD, like the old scripts; use `--outdir Result_random_unitary`
to write alongside the previous results).
