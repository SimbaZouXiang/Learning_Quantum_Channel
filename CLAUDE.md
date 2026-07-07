# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project scope

Research code for **learning noisy quantum channels** (Lindbladian dynamics) via tensor-network parameterised variational models. The canonical library is the **`qcl` package** at the repo root (see README.md for the module map). It was extracted from the former monolith `Learning_Lindbladian/TDME_Trott.py`, which is now a **backward-compatibility shim** that re-exports every `qcl` name — all existing drivers (`import TDME_Trott as tdme`) still work through it. Add new code to the appropriate `qcl` module, never to the shim.

Ignore `scratch/` directories (ad-hoc micro-benchmarks and one-off tests, not a real test suite) and `legacy_generated/` / `legacy_drivers/` (superseded code-generated script clones, kept for reference only).

## Running the code

This is a Compute-Canada / SLURM environment (venv: `~/.virtualenvs/QIP`; `requirements.txt` pins `+computecanada` wheels). Experiments are launched via `job*.sh` SLURM scripts next to their drivers. There is no linter or automated test runner — iterate by running driver scripts directly (`python <script>.py`) or submitting the matching `job_*.sh`. The depolarizing sweep formerly stamped out as `LRU_p*_L*.py` is now `run_depolarizing_sweep.py --p-list ... --L-list ...`.

Trained parameters / losses are dumped as `*.npy` (mostly under `Learning_result/`, `Result_random_unitary/`, `Learning_with_bath/npy_outputs/`); cached training datasets live in `Learning_Lindbladian/Learning_data/`. Data paths are script- or CWD-relative — **don't move data directories, and don't rename `Learning_Lindbladian`/`Learning_with_bath`** (SLURM scripts reach them via `/scratch/simba/*` symlinks).

## Architecture of the qcl package

### Representation layer — Pauli transfer matrices (`qcl.pauli`, `qcl.noise`)

Quantum channels are represented in the **Pauli / vectorised basis** (physical dimension 4: `I,X,Y,Z`), not in the ket basis. This makes channels real-valued and turns noise into a diagonal operation on coefficients.

- Precomputed constants in `qcl.pauli`: `_PAULI_BASIS` (2,2,4), `_PAULI_BASIS_2SITE` (4,4,16), the dephasing transfer matrices `_DEPHASING_TM_{X,Y,Z}`, `_DIR_{X,Y,Z}`, `_EYE4`, `_TRACE_VEC_PAULI`. Built once at import; don't recompute per call.
- `unitary_to_transfer_matrix_{single,two}_site` convert a unitary `U` into its Pauli transfer matrix via `einsum('ab,bci,cd,daj->ij', U, Ps, U†, Ps)/d`.
- `construct_SU4_from_input(params)` takes 16 real parameters and returns a valid SU(4) via QR with phase-normalised diagonal. **All trainable unitaries flow through this.**
- Noise (`qcl.noise`): `depolarization_noise_transfer_matrix(p)` and `dephasing_noise_transfer_matrix_{X,Y,Z}(p)`. `noise_type` arguments select the combination: `"all"`, `"depolarizing"`, `"dephasing"`, `"none"`.

### Two parallel execution paths for the circuit

Every "forward" operation exists in two flavours — keep this split in mind whenever editing:

1. **MPO-building path** (`qcl.mpo`) — `Build_QMLM_MPO` / `Apply_{one,two}_site_layer`. Builds a quimb MPO and merges it with an input MPS via `mpo | mps`; the uncontracted TN goes to `tensor_network_distance`. Simple but scales *exponentially* with depth.
2. **Layer-by-layer MPS path** (`qcl.evolve`) — `Pauli_MPS_after_QMLM_output_only` / `evol_two_site` / `two_site_layer` / `one_site_layer`, and `qcl.tdme.Pauli_MPS_after_TDME_output_only`. Applies each layer directly to a running MPS with `qtn.tensor_network_1d_compress(method="direct", cutoff_mode="rsum2")` bond capping. Cost O(N·D³). Exposed as `QMLM.forward_compressed` and via `use_compressed=True` on the training loops (default for TDME; strongly preferred for deep circuits).

### Index / reindex convention

- Input MPS carries indices `input0, input1, …`; target/output MPS carries `k0, k1, …`. After a forward pass, callers reindex `input{i} → k{i}` before `tensor_network_distance`. Search for `reindex_({f'input{i}': f'k{i}'})` and match it exactly in any new forward variant.
- MPO layers use `k{site}` / `b{site}` internally and are reindexed to deterministic strings (`1site_{form}_L{layer}_S{site}`, `2site_{eo}_L{layer}_S{site}`). New layer types need unique, deterministic index names — random UUIDs break contraction-path caching.

### Models

- `qcl.models.QMLM(N, T)` — **trainable student**. `params` (T, N, 16) complex through `construct_SU4_from_input`, plus per-layer-per-site `p_depolar`, `p_dephase{X,Y,Z}`.
- `qcl.models.QMLM_output_only(N, T)` — **non-trainable teacher** for synthetic data.
- `qcl.bath.QMLM_with_bath_output_only(N, L)` — 2N-site system-bath teacher (data at even sites, bath at odd); bath is partial-traced (`partial_trace_bath`) before comparison.
- `qcl.tdme.TDME(N, T, mu, gamma, J)` — Trotterised time-dependent master equation for a fermionic chain.

### Training / evaluation entry points (`qcl.training`)

- `Learning_MPO` / `Learning_MPO_scheduler` — fit QMLM to a depolarizing teacher (weight-1 Pauli inputs by default; `input_pauli_weight` / `weight_1_pauli_strings` select other bases).
- `Learning_MPO_dephasing_noise_only{,_scheduler}` — dephasing-only teacher.
- `Learning_TDME_scheduler` — fit QMLM to a TDME teacher; loads cached data from `data_dir` via `qcl.io.load_mps` or generates in-process; checkpoints via `LEARNING_TDME_CKPT_DIR` (default CWD-relative `Learning_result`).
- `Learning_QMLM_with_bath_scheduler` — fit QMLM to the system-bath teacher.
- `Testing_TDME_Trotterization{,_parallel}` — Trotter-error benchmarks.

Training loops follow a fixed shape: `epochs` main loop + `epochs // 5` (or `// 4`) fine-tune loop with loosened noise clamp; noise params hard-clamped after every step — `clamp_value = min(strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)`.

Robustness guarantees (added 2026-07-07):
- Every optimizer step goes through `qcl.training._finite_grad_step`, which skips the update (with a WARNING) if any gradient is non-finite — SVD backward can produce NaN grads on near-degenerate singular values, and one poisoned step used to make the parameters permanently NaN and crash everything downstream. A run that prints many of these warnings is not learning; treat it as a failed grid point.
- Testing loops skip samples whose teacher OR student forward fails, and report `testing_loss = NaN` (never `0.0`) when all samples fail — a saved loss of exactly 0.0 in old data may mean "all testing samples failed", not "perfect fit".

### Numerical stability (`qcl.backend`)

- `torch.linalg.svd` is **globally monkey-patched** (scipy `gesvd` fallback + degeneracy jitter). Importing `qcl` or the shim applies it; don't remove or reorder — deep circuits blow up in quimb's compression otherwise.
- A single global `cotengra.ReusableHyperOptimizer` (`qcl.backend.opt`) backs every `tensor_network_distance` call. Never construct a fresh optimiser per call — you lose the path cache and slow contraction by orders of magnitude.

### Parallelism conventions

- Data generation (`qcl.datagen`, `Testing_*_parallel`) uses `ProcessPoolExecutor` with `mp.get_context('spawn')`; workers call `torch.set_num_threads(1)`.
- Training is single-process; per-sample losses go into a Python list combined via `torch.stack(losses).mean()`. Do not parallelise the loss loop with processes — quimb tensors contain unpicklable lambdas; ThreadPoolExecutor is ~2× slower due to the GIL.
- For TDME data generation, `construct_TDME_unitary` and `construct_jump_matrices` are called **once** and passed to every worker — don't regress to per-sample `expm` calls.

### When adding new code

- Put it in the right `qcl` module and, if it's public API, re-export it from `qcl/__init__.py` (and from the `TDME_Trott.py` shim only if legacy drivers need it).
- Feed new trainable unitaries through `construct_SU4_from_input`.
- Use `qcl.backend.tensor_network_distance` (path cache included).
- Maintain the `input{i} → k{i}` reindex pattern at every forward/distance boundary.
- Cast both operands to `complex128` immediately before `tensor_network_distance` — mixed-dtype tensordots crash quimb.
