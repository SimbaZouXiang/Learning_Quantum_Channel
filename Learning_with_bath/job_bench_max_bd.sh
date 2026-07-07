#!/bin/bash
# Run inside debugjob:  debugjob -n 16 ./job_bench_max_bd.sh
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

LOG=slurm_outputs/bench_max_bd_$(date +%Y%m%d-%H%M%S).log
echo "logging to $LOG"
{
  for TL in 3 5; do
    for BD in 16 64 256; do
      echo "=================================================================="
      echo "BENCH  N=10  T=L=$TL  max_bd=$BD"
      echo "=================================================================="
      BENCH_N=10 BENCH_TL=$TL BENCH_MAX_BD=$BD BENCH_G=0.20 \
        BENCH_WARM=1 BENCH_ITER=4 \
        python -u bench_max_bd.py
    done
  done
} 2>&1 | tee "$LOG"
