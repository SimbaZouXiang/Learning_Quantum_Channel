#!/bin/bash
set -euo pipefail
# debugjob inherits login-shell CPU rlimit which kills the Python process
# after a few minutes. Lift it for this allocation.
ulimit -t unlimited || true
ulimit -v unlimited || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
python -u run_bath_N6.py 2>&1 | tee slurm_outputs/debugjob_bathN6.log
