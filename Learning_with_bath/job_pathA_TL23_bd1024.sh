#!/bin/bash
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-16}
LOG=slurm_outputs/pathA_TL23_bd1024_$(date +%H%M%S).log
for TL in 2 3; do
    echo "=================================================================="
    echo "TL=$TL Path A bd=1024 cheat"
    echo "=================================================================="
    PATHA_TL=$TL PATHA_MAX_BD=1024 PATHA_N_TEST=10 python -u cheat_g0_pathA.py
done 2>&1 | tee "$LOG"
