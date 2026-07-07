#!/bin/bash
# Packed TL=2 in debugjob: 11 workers × 8 cores each, single batch (88 cores used).
# Per-worker CPU rlimit budget at 8 cores: 5400/8 = 675 s wall. Per-epoch at 8 cores
# estimated ~5-6 s, so 100+20 epochs ≈ 660 s — fits with thin margin.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=8
N_PARALLEL=11
N_G=11
LOGDIR=slurm_outputs/packed_TL2_debug11x8_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[TL2 debug 11×8] logs in $LOGDIR" | tee "$LOGDIR/driver.log"

PIDS=()
for offset in $(seq 0 $((N_PARALLEL-1))); do
    START=$((offset * CORES_PER_TASK))
    END=$((START + CORES_PER_TASK - 1))
    TASK_LOG=$LOGDIR/worker_g${offset}.log
    (
        export OMP_NUM_THREADS=$CORES_PER_TASK
        export MKL_NUM_THREADS=$CORES_PER_TASK
        export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
        export GOMP_CPU_AFFINITY="$START-$END"
        export OMP_PROC_BIND=close
        export OMP_PLACES="{$START}:$CORES_PER_TASK"
        export BATH_N=10
        export BATH_TL_FILTER=2
        export SLURM_ARRAY_TASK_ID=$offset
        taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
        echo "[TL2 debug 11×8] worker g_idx=$offset exit=$?" | tee -a "$LOGDIR/driver.log"
    ) &
    PIDS+=($!)
    echo "[TL2 debug 11×8]   launched g_idx=$offset on cores $START-$END" | tee -a "$LOGDIR/driver.log"
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[TL2 debug 11×8] all done" | tee -a "$LOGDIR/driver.log"

echo "=== Summary ===" | tee -a "$LOGDIR/driver.log"
for IDX in $(seq 0 $((N_G-1))); do
    LOG=$LOGDIR/worker_g${IDX}.log
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then
        echo "  g_idx=$IDX: ✓ full save" | tee -a "$LOGDIR/driver.log"
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then
        echo "  g_idx=$IDX: △ post-train save only" | tee -a "$LOGDIR/driver.log"
    else
        echo "  g_idx=$IDX: ✗ no save" | tee -a "$LOGDIR/driver.log"
    fi
done
