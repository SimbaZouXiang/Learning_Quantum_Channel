#!/bin/bash
# debugjob 1 ./job_packed_TL2_debugfit.sh
# TL=2 packed in debugjob with BATH_EPOCHS=50 (60 total) — fits in 5400 cpu-sec/proc
# rlimit. Save-before-testing patch ensures training data persists.
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
LOGDIR=slurm_outputs/packed_TL2_debugfit_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[TL2 debugfit] logs in $LOGDIR" | tee "$LOGDIR/driver.log"

for BATCH_START in 0 6; do
    echo "[TL2 debugfit] batch starting at g_idx=$BATCH_START" | tee -a "$LOGDIR/driver.log"
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
            export BATH_EPOCHS=50  # 50 main + 10 fine-tune fits in 5400 cpu-sec/proc rlimit
            export SLURM_ARRAY_TASK_ID=$IDX
            taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
            echo "[TL2 debugfit] worker g_idx=$IDX exit=$?" | tee -a "$LOGDIR/driver.log"
        ) &
        PIDS+=($!)
        echo "[TL2 debugfit]   launched g_idx=$IDX on cores $START-$END" | tee -a "$LOGDIR/driver.log"
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[TL2 debugfit] batch done" | tee -a "$LOGDIR/driver.log"
done

echo "=== Summary ===" | tee -a "$LOGDIR/driver.log"
for IDX in $(seq 0 $((N_G-1))); do
    LOG=$LOGDIR/worker_g${IDX}.log
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then
        echo "  g_idx=$IDX: ✓ full save" | tee -a "$LOGDIR/driver.log"
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then
        echo "  g_idx=$IDX: △ post-train save only (testing block didn't finish)" | tee -a "$LOGDIR/driver.log"
    else
        echo "  g_idx=$IDX: ✗ no save" | tee -a "$LOGDIR/driver.log"
    fi
done
