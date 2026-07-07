#!/bin/bash
#SBATCH --job-name=debug-trott1
#SBATCH --output=slurm-debug-trott1-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian
module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate

pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"

export SLURM_ARRAY_TASK_ID=5  # t=0.8, gamma_name=10
echo "Debug first-order array task starting on $SLURMD_NODENAME at $(date)"
python -u run_trotterization_first_order_one.py
echo "Done at $(date)"
