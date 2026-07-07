#!/bin/bash
#SBATCH --job-name=trace-smoke
#SBATCH --output=slurm-trace-smoke-%j.out
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian/tdme_save_params
module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate
export OMP_NUM_THREADS=192
export MKL_NUM_THREADS=192
export OPENBLAS_NUM_THREADS=192
export NUMBA_NUM_THREADS=192
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
# Smoke: just one (t, gamma) cell to measure realistic timing
export TRACE_T="1.0"
export TRACE_GAMMA="0.1"
echo "Start: $(date)  Node: $SLURMD_NODENAME"
python -u run_trace_norm.py
echo "End:   $(date)"
