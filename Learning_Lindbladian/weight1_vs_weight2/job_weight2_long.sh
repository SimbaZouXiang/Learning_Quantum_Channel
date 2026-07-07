#!/bin/bash
#SBATCH --job-name=w2-long-N8
#SBATCH --output=slurm-w2long-N8-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian/weight1_vs_weight2

module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate

pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
export LEARNING_TDME_CKPT_DIR=/scratch/simba/Learning_Lindbladian/weight1_vs_weight2/results

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
python -u run_weight2_long.py
echo "End: $(date)"
