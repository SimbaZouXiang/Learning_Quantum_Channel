#!/bin/bash
# TL=3 only, 200 epochs, 40 seeds × 4 g = 160 tasks. 1 core per worker
# (160 cores used out of 192), single batch, expected wall ~30 min.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=1
SEEDS=$(seq 200 239)
GIDX=(1 2 3 4)
LOGDIR=slurm_outputs/multiseed_200ep_TL3_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[200ep TL=3] logs $LOGDIR" | tee "$LOGDIR/driver.log"

PIDS=()
slot=0
for SEED in $SEEDS; do
    for G in "${GIDX[@]}"; do
        START=$((slot * CORES_PER_TASK))
        TASK_LOG=$LOGDIR/worker_s${SEED}_g${G}.log
        (
            export OMP_NUM_THREADS=$CORES_PER_TASK
            export MKL_NUM_THREADS=$CORES_PER_TASK
            export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
            export GOMP_CPU_AFFINITY="$START"
            export OMP_PROC_BIND=close
            export OMP_PLACES="{$START}:$CORES_PER_TASK"
            export BATH_N=10
            export BATH_TL_FILTER=3
            export BATH_EPOCHS=200
            export BATH_SEED_OFFSET=$SEED
            export BATH_FILE_EXTRA_SUFFIX=_e200
            export SLURM_ARRAY_TASK_ID=$G
            taskset -c "$START" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
        ) &
        PIDS+=($!)
        slot=$((slot + 1))
    done
done
echo "[200ep TL=3] launched ${#PIDS[@]} workers using $slot cores" | tee -a "$LOGDIR/driver.log"
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[200ep TL=3] done" | tee -a "$LOGDIR/driver.log"

ok=0; partial=0; bad=0
for LOG in "$LOGDIR"/worker_*.log; do
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then ok=$((ok+1))
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then partial=$((partial+1))
    else bad=$((bad+1)); fi
done
echo "[200ep TL=3] ✓ full save: $ok  △ post-train only: $partial  ✗ no save: $bad" \
    | tee -a "$LOGDIR/driver.log"
