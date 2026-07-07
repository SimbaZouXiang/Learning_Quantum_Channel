#!/bin/bash
# DJ2: TL=4 + TL=5 multi-seed extension (seeds 105..119 = 15 additional seeds).
# 150 tasks per depth × 2 depths = 300 tasks total.
# 1 core per worker. Each depth fits in 192 cores, so we run them as two
# sequential batches: batch1 = all TL=4, batch2 = all TL=5. Estimated wall:
# TL=4 ~24 min + TL=5 ~32 min ≈ 56 min, just inside the 60-min debugjob cap.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=1
SEEDS=$(seq 105 119)
GIDX=(1 2 3 4 5 6 7 8 9 10)
LOGDIR=slurm_outputs/multiseed_DJ2_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[DJ2] logs $LOGDIR" | tee "$LOGDIR/driver.log"

run_batch() {
    local TL=$1
    echo "[DJ2] -- batch TL=$TL starting --" | tee -a "$LOGDIR/driver.log"
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
                export GOMP_CPU_AFFINITY="$START"
                export OMP_PROC_BIND=close
                export OMP_PLACES="{$START}:$CORES_PER_TASK"
                export BATH_N=10
                export BATH_TL_FILTER=$TL
                export BATH_EPOCHS=80
                export BATH_SEED_OFFSET=$SEED
                export SLURM_ARRAY_TASK_ID=$G
                taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
            ) &
            PIDS+=($!)
            slot=$((slot + 1))
        done
    done
    echo "[DJ2] TL=$TL launched ${#PIDS[@]} workers, waiting..." \
        | tee -a "$LOGDIR/driver.log"
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[DJ2] TL=$TL batch done" | tee -a "$LOGDIR/driver.log"
}

run_batch 4
run_batch 5

ok=0; partial=0; bad=0
for LOG in "$LOGDIR"/worker_*.log; do
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then ok=$((ok+1))
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then partial=$((partial+1))
    else bad=$((bad+1)); fi
done
echo "[DJ2] ✓ full save: $ok  △ post-train only: $partial  ✗ no save: $bad" \
    | tee -a "$LOGDIR/driver.log"
