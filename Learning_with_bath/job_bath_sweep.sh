#!/bin/bash
#SBATCH --job-name=bath-sweep
#SBATCH --output=slurm_outputs/slurm-bath-sweep-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --array=0-17
# The array covers 3 depths × 6 couplings = 18 combinations.
# --cpus-per-task=192 requests a whole Trillium compute node (192 cores).
# To cap simultaneous tasks at K (useful under fairshare pressure), change the
# array line to `--array=0-17%K`.

set -euo pipefail
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs

source /home/simba/.virtualenvs/QIP/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

# NUMBA cache must live on the compute node (login /home is read-only there).
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

# Default N=10; override by pre-exporting BATH_N before sbatch, or editing here.
export BATH_N=${BATH_N:-10}

python -u run_bath_sweep.py
