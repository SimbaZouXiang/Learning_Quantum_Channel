#!/bin/bash
#SBATCH --job-name=bath-TL2
#SBATCH --output=slurm_outputs/slurm-bath-TL2-%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --array=0-6
# T_L=2, max_bd=256. Measured wallclock ~3-4h per task on a 192-cpu node;
# 8h cap leaves comfortable margin for the testing block.
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
export BATH_TL_FILTER=2

python -u run_bath_sweep.py
