#!/bin/bash
#SBATCH --job-name=smoke-shard
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=debug
#SBATCH --mail-user=xiang.zou@mail.utoronto.ca
#SBATCH --mail-type=END
echo "Job ID: $SLURM_JOB_ID"
echo "Starting run at: `date`"
module load python/3.12.4
source ~/.virtualenvs/QIP/bin/activate

pkill -9 -u $USER -f "TDME_Trott" 2>/dev/null || true
sleep 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}
export MALLOC_ARENA_MAX=2

# Reuse the shard driver but override params for a tiny smoke test.
python -u - <<'PY'
import os, sys, time
SCRIPT_DIR = "/scratch/simba/Learning_Lindbladian"
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
for p in (PARENT_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Override sys.argv before importing the driver's main() — sets target_time/gammas.
sys.argv = ["driver_shard.py", "--target_time", "0.2", "--gammas", "0", "--workers", "1"]

# Patch the driver's main() to use a smaller problem.
import driver_shard
_orig_main = driver_shard.main
def small_main():
    # Hot-patch the parameters before calling the real main flow.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_time", type=float, required=True)
    ap.add_argument("--gammas", type=str, required=True)
    ap.add_argument("--workers", type=int, default=None)
    cli = ap.parse_args()
    target_time  = cli.target_time
    gamma_names  = [int(g) for g in cli.gammas.split(",") if g.strip()]
    N, T = 10, 3
    epochs = 5
    threads = 32
    fut_args = []
    for g in gamma_names:
        fut_args.append((target_time, g, N, T, 3, 1, 1, epochs, 0.05,
                         False, True, "dephasing", False, False, 64, threads))
    print(f"smoke shard: target={target_time} gammas={gamma_names} epochs={epochs} N={N}", flush=True)
    t0 = time.time()
    for args in fut_args:
        driver_shard._worker(*args)
    print(f"smoke shard total wall = {time.time()-t0:.1f}s", flush=True)
small_main()
PY
echo "Finished run at: `date`  (exit=$?)"
