#!/bin/bash
# 200-epoch multi-seed chunk for TL=2 + TL=3 at g in {0.05, 0.10, 0.15, 0.20}.
# Each invocation runs CHUNK_SIZE seeds × 4 g × 2 depths in one debugjob.
# Env vars:
#   CHUNK_START   first seed offset (default 200, to leave 100-199 free)
#   CHUNK_SIZE    number of seeds to do this run (default 50)
#
# Layout: 2 cores/worker, ~10 min/task TL=2, ~20 min/task TL=3.
# CHUNK_SIZE=50 → 200 TL=2 tasks (3 batches × 10 min ≈ 30 min) then 200 TL=3
# tasks (3 batches × 20 min ≈ 60 min). Total ~90 min — over 1h cap, so we
# actually need to split between depths or use larger chunk size carefully.
#
# Default CHUNK_SIZE=25 → 100 TL=2 (~20 min) + 100 TL=3 (~40 min) = ~60 min,
# fits in one debugjob.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CHUNK_START=${CHUNK_START:-200}
CHUNK_SIZE=${CHUNK_SIZE:-25}
CORES_PER_TASK=${CORES_PER_TASK:-2}
SEEDS=$(seq "$CHUNK_START" $((CHUNK_START + CHUNK_SIZE - 1)))
GIDX=(1 2 3 4)
LOGDIR=slurm_outputs/multiseed_200ep_${CHUNK_START}-$((CHUNK_START + CHUNK_SIZE - 1))_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[200ep chunk] seeds $CHUNK_START..$((CHUNK_START + CHUNK_SIZE - 1))  cores=$CORES_PER_TASK  logs=$LOGDIR" \
    | tee "$LOGDIR/driver.log"

run_batch() {
    local TL=$1
    echo "[200ep TL=$TL] -- batch starting --" | tee -a "$LOGDIR/driver.log"
    PIDS=()
    slot=0
    for SEED in $SEEDS; do
        for G in "${GIDX[@]}"; do
            START=$((slot * CORES_PER_TASK))
            END=$((START + CORES_PER_TASK - 1))
            TASK_LOG=$LOGDIR/worker_TL${TL}_s${SEED}_g${G}.log
            (
                export OMP_NUM_THREADS=$CORES_PER_TASK
                export MKL_NUM_THREADS=$CORES_PER_TASK
                export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
                export GOMP_CPU_AFFINITY="$START-$END"
                export OMP_PROC_BIND=close
                export OMP_PLACES="{$START}:$CORES_PER_TASK"
                export BATH_N=10
                export BATH_TL_FILTER=$TL
                export BATH_EPOCHS=200
                export BATH_SEED_OFFSET=$SEED
                export BATH_FILE_EXTRA_SUFFIX=_e200
                export SLURM_ARRAY_TASK_ID=$G
                taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
            ) &
            PIDS+=($!)
            slot=$((slot + 1))
        done
    done
    echo "[200ep TL=$TL] launched ${#PIDS[@]} workers ($((slot * CORES_PER_TASK)) cores)" \
        | tee -a "$LOGDIR/driver.log"
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[200ep TL=$TL] batch done" | tee -a "$LOGDIR/driver.log"
}

run_batch 2
run_batch 3

ok=0; partial=0; bad=0
for LOG in "$LOGDIR"/worker_*.log; do
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then ok=$((ok+1))
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then partial=$((partial+1))
    else bad=$((bad+1)); fi
done
echo "[200ep chunk] ✓ full save: $ok  △ post-train only: $partial  ✗ no save: $bad" \
    | tee -a "$LOGDIR/driver.log"
