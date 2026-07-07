#!/bin/bash
# debugjob -n 32 ./job_cheat_g0.sh
# Cheat-init TL=3, 4, 5 at g=0 and save in sweep .npy format.
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

LOG=slurm_outputs/cheat_g0_$(date +%Y%m%d-%H%M%S).log
echo "logging to $LOG"
CHEAT_MAX_BD=64 python -u cheat_g0_TL345.py 2>&1 | tee "$LOG"
