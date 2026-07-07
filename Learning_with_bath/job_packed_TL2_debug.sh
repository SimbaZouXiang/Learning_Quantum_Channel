#!/bin/bash
# Quick debugjob preview for TL=2 with new config (max_bd=128, use_compressed=True).
#   debugjob 1 ./job_packed_TL2_debug.sh
# Runs only BATH_EPOCHS=20 (24 total epochs) to fit inside the 5400 cpu-sec/process
# rlimit. 6 workers per batch × 32 cores, 2 batches cover all 11 g values.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=32
N_PARALLEL=6
N_G=11
LOGDIR=slurm_outputs/packed_TL2_preview_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[TL2 preview] logs in $LOGDIR" | tee "$LOGDIR/driver.log"

for BATCH_START in 0 6; do
    echo "[TL2 preview] batch starting at g_idx=$BATCH_START" | tee -a "$LOGDIR/driver.log"
    PIDS=()
    for offset in $(seq 0 $((N_PARALLEL-1))); do
        IDX=$((BATCH_START + offset))
        [ "$IDX" -ge "$N_G" ] && continue
        START=$((offset * CORES_PER_TASK))
        END=$((START + CORES_PER_TASK - 1))
        TASK_LOG=$LOGDIR/worker_g${IDX}.log
        (
            export OMP_NUM_THREADS=$CORES_PER_TASK
            export MKL_NUM_THREADS=$CORES_PER_TASK
            export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
            export GOMP_CPU_AFFINITY="$START-$END"
            export OMP_PROC_BIND=close
            export OMP_PLACES="{$START}:$CORES_PER_TASK"
            export BATH_N=10
            export BATH_TL_FILTER=2
            export BATH_EPOCHS=20   # fits in debugjob's 5400 cpu-sec/proc rlimit
            export SLURM_ARRAY_TASK_ID=$IDX
            taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
            echo "[TL2 preview] worker g_idx=$IDX exit=$?" | tee -a "$LOGDIR/driver.log"
        ) &
        PIDS+=($!)
        echo "[TL2 preview]   launched g_idx=$IDX on cores $START-$END" | tee -a "$LOGDIR/driver.log"
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[TL2 preview] batch done" | tee -a "$LOGDIR/driver.log"
done

echo "=== Summary ===" | tee -a "$LOGDIR/driver.log"
for IDX in $(seq 0 $((N_G-1))); do
    LOG=$LOGDIR/worker_g${IDX}.log
    [ -f "$LOG" ] || continue
    LAST=$(grep -E "Epoch [0-9]+, Loss:" "$LOG" | tail -1 | sed -E 's/.*Loss: ([0-9.e+-]+) .*/\1/')
    SAVED=$(grep -c "post-train save\|saved.*\.npy" "$LOG")
    echo "  g_idx=$IDX  last_loss=$LAST  saves=$SAVED" | tee -a "$LOGDIR/driver.log"
done
