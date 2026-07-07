#!/bin/bash
#SBATCH --job-name=depol_sweep
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mail-user=xiang.zou@mail.utoronto.ca
#SBATCH --mail-type=END
# ---------------------------------------------------------------------
# Usage:
#   sbatch job_depolarizing_sweep.sh --p-list 0 2 4 --L-list 3 4 5
#   sbatch job_depolarizing_sweep.sh --p-list 6 8 10 --L-list 6 7 --threads-per-worker 8
# All arguments are forwarded to run_depolarizing_sweep.py.
# Replaces the generated job_pX_LY.sh / LRU_pX_LY.py pairs (see legacy_generated/).
# ---------------------------------------------------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
module load python/3.12.4
source ~/.virtualenvs/QIP/bin/activate

# Kill any orphaned python workers from previous runs
pkill -9 -u $USER -f "run_depolarizing_sweep" 2>/dev/null || true
sleep 1

# Safe thread defaults; each worker re-pins itself before importing torch.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}

cd "$SLURM_SUBMIT_DIR"
python run_depolarizing_sweep.py "$@"
# ---------------------------------------------------------------------
echo "Finished run at: `date`"
