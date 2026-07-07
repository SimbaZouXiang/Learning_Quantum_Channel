"""Generic (T=L, g) sweep driver for QMLM-with-bath at fixed N.

Each invocation runs ONE combination, picked by `$SLURM_ARRAY_TASK_ID`.

  task_idx = t_l_idx * len(COUPLINGS) + g_idx
    t_l_idx ∈ [0, len(T_L_VALUES))
    g_idx   ∈ [0, len(COUPLINGS))

`max_bd` is per-T_L: shallower depths can afford a larger target bond.
Output filenames embed the bond cap (e.g. `..._bd256.npy`) so runs at
different bond caps don't clobber each other.

Set `BATH_TL_FILTER` (e.g. `"3"` or `"3,4"`) to restrict the sweep to a subset
of T_L values — convenient for splitting the array across SLURM jobs with
different walltimes per depth.
"""
import os
import sys
import time
import warnings
import resource

# Some interactive allocations (notably `debugjob`) inherit the login shell's
# RLIMIT_CPU = 3600s, which kills a many-thread process whose cpu-seconds
# (= cores * wall) climb past 1 hour. Raise it as high as we're allowed.
for _lim in (resource.RLIMIT_CPU, resource.RLIMIT_AS):
    try:
        soft, hard = resource.getrlimit(_lim)
        resource.setrlimit(_lim, (hard, hard))
    except (ValueError, OSError, resource.error):
        pass

warnings.filterwarnings("ignore", message="Casting complex")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Learning_Lindbladian"))
NPY_DIR = os.path.join(SCRIPT_DIR, "npy_outputs")
os.makedirs(NPY_DIR, exist_ok=True)

import numpy as np
import torch

import TDME_Trott as tdme

# ===== sweep configuration =====
N            = int(os.environ.get("BATH_N", 10))
T_L_VALUES   = [2, 3, 4, 5]
COUPLINGS    = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# New high-g values appended (indices 7..10) so the previously-submitted
# array=0-6 jobs still map to g=0.0..0.30 unchanged.
# Per-T_L bond cap. TL2 keeps the Path-A max_bd=256 setting that already
# produced clean results. TL3/4/5 use max_bd=64 paired with the compressed
# layer-by-layer path (use_compressed=True below) — that combination gives
# higher-fidelity targets than the legacy bd=16 Path A runs and runs ~5x
# faster than Path A bd=64.
MAX_BD_BY_TL = {2: 256, 3: 64, 4: 64, 5: 64}
# Switch to the compressed path with proper truncation (the apply_mpo_to_mps_
# compressed bug fix is in Learning_Lindbladian/TDME_Trott.py). All depths now
# use the compressed path for consistency; TL=2 was previously bd=256 Path A
# but the resulting loss curve was qualitatively offset from the deeper runs.
USE_COMPRESSED_BY_TL = {2: True, 3: True, 4: True, 5: True}
DEPOL_STRENGTH = 0.1
# Honour an env override for quick correctness tests inside a debugjob
# (whose 5400 s/process CPU rlimit kills the default 100-epoch run).
EPOCHS       = int(os.environ.get("BATH_EPOCHS", 100))
LR           = 0.05
MAX_ERR      = 1e-6
# ===============================


def main():
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if idx_str is None:
        raise SystemExit("SLURM_ARRAY_TASK_ID not set; run as a SLURM array job.")
    idx = int(idx_str)

    # Optional filter: restrict to a subset of T_L values (per-depth job split).
    tl_filter_env = os.environ.get("BATH_TL_FILTER", "")
    if tl_filter_env:
        tl_subset = [int(x) for x in tl_filter_env.split(",") if x.strip()]
        active_tls = [tl for tl in T_L_VALUES if tl in tl_subset]
        if not active_tls:
            raise SystemExit(f"BATH_TL_FILTER={tl_filter_env!r} matched no T_L values "
                             f"(allowed {T_L_VALUES})")
    else:
        active_tls = list(T_L_VALUES)

    n_g, n_tl = len(COUPLINGS), len(active_tls)
    total = n_tl * n_g
    if not (0 <= idx < total):
        raise SystemExit(f"array index {idx} outside [0, {total}) for active T_L={active_tls}")
    t_l_idx, g_idx = divmod(idx, n_g)
    T_L = active_tls[t_l_idx]
    g = COUPLINGS[g_idx]
    max_bd = MAX_BD_BY_TL[T_L]
    use_compressed = USE_COMPRESSED_BY_TL[T_L]
    print(f"=== task {idx}/{total}: N={N}, T=L={T_L}, g={g}, max_bd={max_bd}, "
          f"use_compressed={use_compressed} ===", flush=True)

    # Seed per (T=L, g) pair. Using T_L keeps the teacher's Haar gates
    # identical across couplings at the same depth, so g-trends are
    # apples-to-apples at fixed T_L. Optional BATH_SEED_OFFSET tags an
    # alternate seed for multi-seed studies — files also get a `_s{offset}`
    # suffix in that case so they don't clobber the canonical seed-0 sweep.
    seed_offset = int(os.environ.get("BATH_SEED_OFFSET", 0))
    seed = 100 * T_L + seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))))

    g_tag = f"{int(round(g * 100)):03d}"
    seed_suffix = f"_s{seed_offset}" if seed_offset != 0 else ""
    # Optional extra suffix for sub-experiments that want their own filename
    # namespace (e.g., long-training runs at BATH_EPOCHS=280 vs the canonical
    # BATH_EPOCHS=80). Default empty = back-compat with all existing files.
    extra_suffix = os.environ.get("BATH_FILE_EXTRA_SUFFIX", "")
    prefix = (f"bath_sweep_N{N}_T{T_L}_L{T_L}_g{g_tag}"
              f"_p{int(DEPOL_STRENGTH*100):03d}_bd{max_bd}{seed_suffix}{extra_suffix}")

    def _save_after_training(snap):
        """Persist training artifacts as soon as the 120 epochs finish, so a
        slow/hung testing block doesn't destroy hours of work. Testing-block
        outputs are written separately below if/when that block completes."""
        m = snap['model']
        ll = np.array(snap['learning_loss'])
        np.save(os.path.join(NPY_DIR, f"{prefix}_learning_loss.npy"), ll)
        np.save(os.path.join(NPY_DIR, f"{prefix}_model_param.npy"),
                m.params.detach().numpy())
        np.save(os.path.join(NPY_DIR, f"{prefix}_model_p_depolar.npy"),
                m.p_depolar.detach().numpy())
        print(f"  [post-train save] wrote {prefix}_{{learning_loss,model_param,"
              f"model_p_depolar}}.npy", flush=True)

    t0 = time.time()
    out = tdme.Learning_QMLM_with_bath_scheduler(
        N=N, MPO_layer=T_L, model_to_learn_layer=T_L,
        coupling_strength=g, J_b=1.0,
        depolarizing_strength=DEPOL_STRENGTH,
        epochs=EPOCHS, lr=LR,
        max_bd=max_bd, max_err=MAX_ERR, truncation=True,
        noise_type="depolarizing",
        use_compressed=use_compressed, num_threads=1,
        post_train_callback=_save_after_training,
    )
    elapsed = time.time() - t0
    print(f"=== task {idx}: finished in {elapsed:.1f}s ===", flush=True)

    (model, learning_loss, haar_list, weak_list,
     testing_loss, testing_loss_list,
     model_param, model_p_depolar,
     model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ) = out

    # learning_loss / model_param / model_p_depolar were already written by
    # _save_after_training; rewrite with the post-testing values (functionally
    # identical, just keeps the testing-block .npy companions in sync).
    np.save(os.path.join(NPY_DIR, f"{prefix}_learning_loss.npy"), np.array(learning_loss))
    np.save(os.path.join(NPY_DIR, f"{prefix}_testing_loss.npy"), np.array([testing_loss]))
    np.save(os.path.join(NPY_DIR, f"{prefix}_testing_loss_list.npy"), np.array(testing_loss_list))
    np.save(os.path.join(NPY_DIR, f"{prefix}_model_param.npy"), model_param)
    np.save(os.path.join(NPY_DIR, f"{prefix}_model_p_depolar.npy"), model_p_depolar)
    print(f"saved {prefix}_*.npy to {NPY_DIR}", flush=True)


if __name__ == "__main__":
    main()
