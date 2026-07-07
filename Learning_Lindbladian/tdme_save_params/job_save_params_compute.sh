#!/bin/bash
#SBATCH --job-name=tdme-train-big
#SBATCH --output=slurm-tdme-train-big-%j.out
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian/tdme_save_params
module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate
pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1
# Per-worker BLAS thread budget (driver re-sets these inside the worker too)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
# 24 parallel workers x 8 threads = 192 cores
export TDME_THREADS_PER_WORKER=8
# Phase 2: expensive variants only.  Cheap variants run on debug partition separately.
export TDME_VARIANTS=${TDME_VARIANTS:-w2full,combined,random276}
echo "Start: $(date)  Node: $SLURMD_NODENAME  TDME_THREADS_PER_WORKER=$TDME_THREADS_PER_WORKER  TDME_VARIANTS=$TDME_VARIANTS"
python -u run_save_params.py
echo "End:   $(date)"
