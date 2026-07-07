#!/bin/bash
# debugjob -n 16 ./job_test_fix.sh
set -euo pipefail
ulimit -t unlimited || true
ulimit -v unlimited || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

LOG=slurm_outputs/test_fix_$(date +%Y%m%d-%H%M%S).log
echo "logging to $LOG"
TEST_EPOCHS=10 TEST_MAX_BD=64 python -u test_fix_scheduler.py 2>&1 | tee "$LOG"
