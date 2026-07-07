#!/bin/bash
# Run with: debugjob -n 16 ./job_test_compressed.sh
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

LOG=slurm_outputs/test_compressed_$(date +%Y%m%d-%H%M%S).log
echo "logging to $LOG"
{
  for TL in 3 4; do
    for BD in 16 64; do
      echo "==================================================================="
      echo "TEST  N=10  T=L=$TL  g=0.20  max_bd=$BD"
      echo "==================================================================="
      TEST_N=10 TEST_TL=$TL TEST_G=0.20 TEST_MAX_BD=$BD \
        python -u test_compressed_path.py
    done
  done
} 2>&1 | tee "$LOG"
