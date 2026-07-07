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
export PATHA_TL=5
export PATHA_MAX_BD=1024
export PATHA_N_TEST=3
LOG=slurm_outputs/pathA_TL5_bd1024_$(date +%H%M%S).log
python -u cheat_g0_pathA.py 2>&1 | tee "$LOG"
