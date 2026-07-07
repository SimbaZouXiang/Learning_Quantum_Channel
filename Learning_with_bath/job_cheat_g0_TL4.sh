#!/bin/bash
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-32}
export CHEAT_MAX_BD=64
export CHEAT_TL=4
export CHEAT_N_TEST=30
LOG=slurm_outputs/cheat_g0_TL4_$(date +%H%M%S).log
python -u cheat_g0_TL345.py 2>&1 | tee "$LOG"
