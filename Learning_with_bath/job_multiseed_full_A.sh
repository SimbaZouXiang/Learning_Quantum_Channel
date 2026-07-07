#!/bin/bash
# DJ-A: multi-seed sweep for TL=2 (all g) and TL=3 (g >= 0.25). 80 tasks total.
# 2 cores per worker × 80 workers = 160 cores used. One batch, wall ~10-15 min.
#   debugjob 1 ./job_multiseed_full_A.sh
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=2
LOGDIR=slurm_outputs/multiseed_fullA_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[multiseed A] logs in $LOGDIR" | tee "$LOGDIR/driver.log"

SEED_OFFSETS=(0 1 2 3 4)
# COUPLINGS indices: [0]=0.00, [1]=0.05, [2]=0.10, [3]=0.15, [4]=0.20,
#                    [5]=0.25, [6]=0.30, [7]=0.35, [8]=0.40, [9]=0.45, [10]=0.50
TL2_GIDX=(1 2 3 4 5 6 7 8 9 10)   # all g>0 for TL=2
TL3_GIDX=(5 6 7 8 9 10)           # g >= 0.25 only (lower g already done)

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
        echo "[multiseed A] TL=$TL seed=$SEED g_idx=$GIDX exit=$?" \
            | tee -a "$LOGDIR/driver.log"
    ) &
    PIDS+=($!)
    slot=$((slot + 1))
}

for TL in 2 3; do
    if [ "$TL" -eq 2 ]; then
        GLIST=("${TL2_GIDX[@]}")
    else
        GLIST=("${TL3_GIDX[@]}")
    fi
    for SEED in "${SEED_OFFSETS[@]}"; do
        for GIDX in "${GLIST[@]}"; do
            launch "$TL" "$GIDX" "$SEED"
        done
    done
done

echo "[multiseed A] launched ${#PIDS[@]} workers using $((slot * CORES_PER_TASK)) cores" \
    | tee -a "$LOGDIR/driver.log"
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[multiseed A] all done" | tee -a "$LOGDIR/driver.log"

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
