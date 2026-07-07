#!/bin/bash
#SBATCH --job-name=TDME_t0p2_a
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mail-user=xiang.zou@mail.utoronto.ca
#SBATCH --mail-type=END
# ---------------------------------------------------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
module load python/3.12.4
mkdir -p ~/.virtualenvs
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

python -u driver_shard.py --target_time 0.2 --gammas 0,2,4,6,8
RC=$?
echo "Finished run at: `date`  (exit=$RC)"
exit $RC
