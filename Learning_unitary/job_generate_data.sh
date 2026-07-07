#!/bin/bash
#SBATCH --job-name=gen_mps_data
#SBATCH --time=48:00:00
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

# ── Kill any orphaned python workers from previous runs ──────────────
pkill -9 -u $USER -f "generate_training_data" 2>/dev/null || true
pkill -9 -u $USER -f "TDME_Trott" 2>/dev/null || true
sleep 1

# ── Thread settings ──────────────────────────────────────────────────
# Set safe defaults here; each worker process overrides these with
# threads_per_worker before importing torch/numpy.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}

cd /scratch/simba/Learning_unitary
python generate_training_data.py --workers 6
# ---------------------------------------------------------------------
echo "Finished run at: `date`"
