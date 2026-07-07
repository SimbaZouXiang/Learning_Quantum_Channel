#!/bin/bash
#SBATCH --job-name=auditN6
#SBATCH --output=slurm_outputs/slurm-auditN6-%j.out
#SBATCH --time=01:00:00
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
mkdir -p "$NUMBA_CACHE_DIR"
python -u audit_N6_consistency.py
