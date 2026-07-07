#!/bin/bash
# DJ3: recover TL=5 multi-seed (DJ2's TL=5 batch over-packed at 1 core/worker
# and ran out of debugjob wallclock). Use 4 cores/worker × 48 workers per batch
# for clean NUMA placement; 80 tasks → 2 batches × ~25 min = ~50 min.
# Seeds 105..112 (= 8 additional seeds), bringing TL=5 total from 10 to 18.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=4
BATCH_CAP=48              # 192 / 4
SEEDS=$(seq 105 112)
GIDX=(1 2 3 4 5 6 7 8 9 10)
LOGDIR=slurm_outputs/multiseed_DJ3_TL5_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[DJ3 TL=5] logs $LOGDIR" | tee "$LOGDIR/driver.log"

TASKS=()
for SEED in $SEEDS; do
    for G in "${GIDX[@]}"; do
        TASKS+=("$G $SEED")
    done
done
TOTAL=${#TASKS[@]}
echo "[DJ3 TL=5] total tasks: $TOTAL  batch cap: $BATCH_CAP  cores/worker: $CORES_PER_TASK" \
    | tee -a "$LOGDIR/driver.log"

i=0
batch_num=0
while [ $i -lt $TOTAL ]; do
    batch_num=$((batch_num + 1))
    echo "[DJ3 TL=5] -- batch $batch_num starting at task $i --" | tee -a "$LOGDIR/driver.log"
    PIDS=()
    slot=0
    while [ $i -lt $TOTAL ] && [ $slot -lt $BATCH_CAP ]; do
        IFS=' ' read -r G SEED <<< "${TASKS[$i]}"
        START=$((slot * CORES_PER_TASK))
        END=$((START + CORES_PER_TASK - 1))
        TASK_LOG=$LOGDIR/worker_s${SEED}_g${G}.log
        (
            export OMP_NUM_THREADS=$CORES_PER_TASK
            export MKL_NUM_THREADS=$CORES_PER_TASK
            export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
            export GOMP_CPU_AFFINITY="$START-$END"
            export OMP_PROC_BIND=close
            export OMP_PLACES="{$START}:$CORES_PER_TASK"
            export BATH_N=10
            export BATH_TL_FILTER=5
            export BATH_EPOCHS=80
            export BATH_SEED_OFFSET=$SEED
            export SLURM_ARRAY_TASK_ID=$G
            taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
        ) &
        PIDS+=($!)
        slot=$((slot + 1))
        i=$((i + 1))
    done
    echo "[DJ3 TL=5] batch $batch_num: ${#PIDS[@]} workers, waiting..." | tee -a "$LOGDIR/driver.log"
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[DJ3 TL=5] batch $batch_num done" | tee -a "$LOGDIR/driver.log"
done

ok=0; partial=0; bad=0
for LOG in "$LOGDIR"/worker_*.log; do
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then ok=$((ok+1))
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then partial=$((partial+1))
    else bad=$((bad+1)); fi
done
echo "[DJ3 TL=5] ✓ full save: $ok  △ post-train only: $partial  ✗ no save: $bad" \
    | tee -a "$LOGDIR/driver.log"
