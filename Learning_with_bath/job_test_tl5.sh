#!/bin/bash
set -euo pipefail
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
LOG=slurm_outputs/test_tl5_$(date +%Y%m%d-%H%M%S).log
N_EPOCHS=5 python -u test_tl5_compressed.py 2>&1 | tee "$LOG"
