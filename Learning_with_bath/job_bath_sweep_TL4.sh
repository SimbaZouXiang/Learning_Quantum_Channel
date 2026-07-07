#!/bin/bash
#SBATCH --job-name=bath-TL4
#SBATCH --output=slurm_outputs/slurm-bath-TL4-%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --array=0-10
# T_L=4, max_bd=64, use_compressed=True (compressed-path bug fix in place).
# Benched at ~10s/iter in debugjob (16 cpus); production estimate <90 min/task
# at 192 cpus including a worst-case 4x threading penalty. 10h gives ~6x margin.
set -euo pipefail
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs

source /home/simba/.virtualenvs/QIP/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"
export BATH_N=${BATH_N:-10}
export BATH_TL_FILTER=4

python -u run_bath_sweep.py
