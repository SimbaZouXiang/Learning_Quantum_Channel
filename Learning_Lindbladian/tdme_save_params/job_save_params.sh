#!/bin/bash
#SBATCH --job-name=tdme-train
#SBATCH --output=slurm-tdme-train-%j.out
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian/tdme_save_params
module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate
pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1
# These will be overridden per-worker inside the driver
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
# Threads per worker: set from smoke results.  e.g. 8 -> 24 parallel workers on 192 cores.
export TDME_THREADS_PER_WORKER=${TDME_THREADS_PER_WORKER:-8}
# Phase 1: cheap variants only (each ~17 min/cell, fits debug walltime).
export TDME_VARIANTS=${TDME_VARIANTS:-w1,random24}
echo "Start: $(date)  Node: $SLURMD_NODENAME  TDME_THREADS_PER_WORKER=$TDME_THREADS_PER_WORKER  TDME_VARIANTS=$TDME_VARIANTS"
python -u run_save_params.py
echo "End:   $(date)"
