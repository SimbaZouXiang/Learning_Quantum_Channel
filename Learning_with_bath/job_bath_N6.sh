#!/bin/bash
#SBATCH --job-name=bathN6
#SBATCH --output=slurm_outputs/slurm-bathN6-%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache
export NUMBA_DISABLE_JIT=0
mkdir -p "$NUMBA_CACHE_DIR"
python -u run_bath_N6.py
