#!/bin/bash
# DJ-B: multi-seed sweep for TL=4 and TL=5 (g >= 0.25 only). 60 tasks total.
# 3 cores per worker × 60 workers = 180 cores used. One batch, wall ~20-25 min.
#   debugjob 1 ./job_multiseed_full_B.sh
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=3
LOGDIR=slurm_outputs/multiseed_fullB_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[multiseed B] logs in $LOGDIR" | tee "$LOGDIR/driver.log"

SEED_OFFSETS=(0 1 2 3 4)
TL_GIDX=(5 6 7 8 9 10)   # g >= 0.25 (lower-g points already multi-seeded)

slot=0
PIDS=()
launch() {
    local TL=$1 GIDX=$2 SEED=$3
    local START=$((slot * CORES_PER_TASK))
    local END=$((START + CORES_PER_TASK - 1))
    local TASK_LOG=$LOGDIR/worker_TL${TL}_s${SEED}_g${GIDX}.log
    (
        export OMP_NUM_THREADS=$CORES_PER_TASK
        export MKL_NUM_THREADS=$CORES_PER_TASK
        export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
        export GOMP_CPU_AFFINITY="$START-$END"
        export OMP_PROC_BIND=close
        export OMP_PLACES="{$START}:$CORES_PER_TASK"
        export BATH_N=10
        export BATH_TL_FILTER=$TL
        export BATH_EPOCHS=80
        export BATH_SEED_OFFSET=$((SEED + 100))
        export SLURM_ARRAY_TASK_ID=$GIDX
        taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
        echo "[multiseed B] TL=$TL seed=$SEED g_idx=$GIDX exit=$?" \
            | tee -a "$LOGDIR/driver.log"
    ) &
    PIDS+=($!)
    slot=$((slot + 1))
}

for TL in 4 5; do
    for SEED in "${SEED_OFFSETS[@]}"; do
        for GIDX in "${TL_GIDX[@]}"; do
            launch "$TL" "$GIDX" "$SEED"
        done
    done
done

echo "[multiseed B] launched ${#PIDS[@]} workers using $((slot * CORES_PER_TASK)) cores" \
    | tee -a "$LOGDIR/driver.log"
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[multiseed B] all done" | tee -a "$LOGDIR/driver.log"

echo "=== Summary ===" | tee -a "$LOGDIR/driver.log"
ok=0; partial=0; bad=0
for LOG in "$LOGDIR"/worker_*.log; do
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then
        ok=$((ok+1))
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then
        partial=$((partial+1))
    else
        bad=$((bad+1))
    fi
done
echo "  ✓ full save: $ok   △ post-train only: $partial   ✗ no save: $bad" \
    | tee -a "$LOGDIR/driver.log"
