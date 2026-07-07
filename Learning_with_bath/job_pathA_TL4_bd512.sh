#!/bin/bash
# Path A cheat-init TL=4 bd=512 in debugjob. 16 cores per task; with 5 test
# samples this should fit in 5400 cpu-sec/process rlimit.
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
export PATHA_TL=4
export PATHA_MAX_BD=512
export PATHA_N_TEST=10
LOG=slurm_outputs/pathA_TL4_bd512_$(date +%H%M%S).log
python -u cheat_g0_pathA.py 2>&1 | tee "$LOG"
