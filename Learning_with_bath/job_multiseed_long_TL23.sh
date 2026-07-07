#!/bin/bash
# Long-training multi-seed run for TL=2 and TL=3 at g in {0.05, 0.10, 0.15, 0.20}.
# Uses BATH_EPOCHS=280 (= 280 main + 56 fine-tune = 336 total epochs, vs the
# canonical 80+16=96) to test whether the high seed-to-seed std at small g
# shrinks given more training time. 20 seeds per (T_L, g) point.
#
# Layout: 2 cores/worker, 80 workers per batch, 2 sequential batches
# (TL=2 first then TL=3). Estimated wall: ~11 min + ~28 min = ~40 min.
#
# Files are tagged with `_e280` extra suffix to avoid clobbering the canonical
# BATH_EPOCHS=80 multi-seed sweep:
#   bath_sweep_N10_T{2,3}_L{2,3}_g{tag}_p010_bd{bd}_s{seed}_e280_*.npy
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=2
SEEDS=$(seq 100 119)
GIDX=(1 2 3 4)
LOGDIR=slurm_outputs/multiseed_long_TL23_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[long TL=2/3] BATH_EPOCHS=280  cores/worker=$CORES_PER_TASK  logs=$LOGDIR" \
    | tee "$LOGDIR/driver.log"

run_batch() {
    local TL=$1
    echo "[long TL=$TL] -- batch starting --" | tee -a "$LOGDIR/driver.log"
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
                export BATH_EPOCHS=280
                export BATH_SEED_OFFSET=$SEED
                export BATH_FILE_EXTRA_SUFFIX=_e280
                export SLURM_ARRAY_TASK_ID=$G
                taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
            ) &
            PIDS+=($!)
            slot=$((slot + 1))
        done
    done
    echo "[long TL=$TL] launched ${#PIDS[@]} workers using $((slot * CORES_PER_TASK)) cores" \
        | tee -a "$LOGDIR/driver.log"
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[long TL=$TL] batch done" | tee -a "$LOGDIR/driver.log"
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
echo "[long TL=2/3] ✓ full save: $ok  △ post-train only: $partial  ✗ no save: $bad" \
    | tee -a "$LOGDIR/driver.log"
