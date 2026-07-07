#!/bin/bash
#SBATCH --job-name=mpo-ext-eval
#SBATCH --output=slurm-mpo-ext-eval-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian/mpo_extended_eval
module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate

pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1

export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export OPENBLAS_NUM_THREADS=18
export NUMBA_NUM_THREADS=18
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"

echo "Start: $(date)  Node: $SLURMD_NODENAME"
python -u run_extended_eval.py
echo "End:   $(date)"
