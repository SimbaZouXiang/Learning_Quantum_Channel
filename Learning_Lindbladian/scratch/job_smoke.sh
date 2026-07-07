#!/bin/bash
#SBATCH --job-name=smoke-TDME
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=debug
#SBATCH --mail-user=xiang.zou@mail.utoronto.ca
#SBATCH --mail-type=END
# ---------------------------------------------------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
module load python/3.12.4
source ~/.virtualenvs/QIP/bin/activate

pkill -9 -u $USER -f "Learning_random_unitary" 2>/dev/null || true
pkill -9 -u $USER -f "TDME_Trott" 2>/dev/null || true
sleep 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}
export MALLOC_ARENA_MAX=2

python -u debug_smoke.py
RC=$?
echo "Finished run at: `date`  (exit=$RC)"
exit $RC
