#!/bin/bash
#SBATCH --job-name=trott-array
#SBATCH --output=slurm-trott-array-%A_%a.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --array=0-29
# 3 times (0.8, 1.0, 2.0) x 10 gammas (0,2,4,6,8,10,20,30,40,50) = 30 tasks.
# Each array element runs one (t, gamma) on its own full Trillium node
# with 48 worker processes (MaxRSS ~217 GB observed; 767 GB node has plenty
# of headroom).  6h walltime is ~10x the 21-min Apr-29 (buggy) run, so the
# correct-numerics run should fit comfortably.

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian

module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate

# ── orphan cleanup ────────────────────────────────────────────────────
pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1

# ── thread settings: each worker process uses 1 thread; we run 48
# workers in parallel → 48 cores active out of 192. The remaining cores
# stay idle but the node has to be allocated whole anyway. Bumping
# workers caused OOM on the same problem before (Apr 22, 100 workers,
# MaxRSS 720 GB / 767 GB).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"

echo "Array task $SLURM_ARRAY_TASK_ID on $SLURMD_NODENAME starting $(date)"
python -u run_trotterization_one.py
echo "Array task $SLURM_ARRAY_TASK_ID finished $(date)"
