#!/bin/bash
# Designed to be invoked inside a debugjob:
#   debugjob -n 16 ./job_verify_g0.sh
# (or: source it after `salloc`-ing). Runs the cheat-init verification at
# (N, T=L) triples covering the same depths as the N=10 sweep.
set -euo pipefail
ulimit -t unlimited || true
ulimit -v unlimited || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

LOG=slurm_outputs/verify_g0_$(date +%Y%m%d-%H%M%S).log
echo "logging to $LOG"
{
  for TL in 3 4 5; do
    for BD in 16 64 256; do
      echo "=================================================================="
      echo "VERIFY g=0  N=10  T=L=$TL  max_bd=$BD"
      echo "=================================================================="
      VERIFY_N=10 VERIFY_TL=$TL VERIFY_MAX_BD=$BD \
        python -u verify_g0_perfect.py
    done
  done
} 2>&1 | tee "$LOG"
