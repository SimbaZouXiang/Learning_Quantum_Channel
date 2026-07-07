#!/bin/bash
#SBATCH --job-name=w1-vs-rand
#SBATCH --output=slurm-w1-rand-%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --array=0-17
# 3 N (8,10,12) × 6 L (3..8) = 18 tasks. Each is one full Trillium node;
# inside, the run does 2 trainings (weight-1 + random) sequentially.

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian
module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate

pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1

export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"

# 192 cores / 1 worker = up to 192 threads, but tensor_network_distance
# saturates well below that. 8 threads per BLAS call is a safe default.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=8

echo "Array task $SLURM_ARRAY_TASK_ID on $SLURMD_NODENAME starting $(date)"
python -u run_weight1_vs_random_one.py
echo "Finished $(date)"
