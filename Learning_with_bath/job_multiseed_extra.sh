#!/bin/bash
# Parameterised single-depth multi-seed sweep, sized for one debugjob.
# Required env vars:
#   EXTRA_TL          target depth (2, 3, 4, or 5)
#   EXTRA_SEED_START  first seed offset to add (e.g. 105)
#   EXTRA_SEED_END    last seed offset (inclusive, e.g. 119)
#   EXTRA_CORES       cores per worker (default 1)
# All g>0 indices (1..10) are swept by default; override via EXTRA_GIDX="1 2 3"
#
# Filenames are tagged `_s{offset}` so this never clobbers the canonical
# single-seed sweep on disk or earlier multi-seed runs.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

: "${EXTRA_TL:?must set EXTRA_TL}"
: "${EXTRA_SEED_START:?must set EXTRA_SEED_START}"
: "${EXTRA_SEED_END:?must set EXTRA_SEED_END}"
CORES_PER_TASK=${EXTRA_CORES:-1}
GIDX_DEFAULT="1 2 3 4 5 6 7 8 9 10"
GIDX_LIST=(${EXTRA_GIDX:-$GIDX_DEFAULT})

LOGDIR=slurm_outputs/multiseed_extra_TL${EXTRA_TL}_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[multiseed extra] TL=$EXTRA_TL  seeds $EXTRA_SEED_START..$EXTRA_SEED_END  cores=$CORES_PER_TASK  logs=$LOGDIR" \
    | tee "$LOGDIR/driver.log"

slot=0
PIDS=()
for SEED in $(seq "$EXTRA_SEED_START" "$EXTRA_SEED_END"); do
    for GIDX in "${GIDX_LIST[@]}"; do
        START=$((slot * CORES_PER_TASK))
        END=$((START + CORES_PER_TASK - 1))
        TASK_LOG=$LOGDIR/worker_s${SEED}_g${GIDX}.log
        (
            export OMP_NUM_THREADS=$CORES_PER_TASK
            export MKL_NUM_THREADS=$CORES_PER_TASK
            export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
            if [ "$CORES_PER_TASK" -gt 1 ]; then
                export GOMP_CPU_AFFINITY="$START-$END"
            else
                export GOMP_CPU_AFFINITY="$START"
            fi
            export OMP_PROC_BIND=close
            export OMP_PLACES="{$START}:$CORES_PER_TASK"
            export BATH_N=10
            export BATH_TL_FILTER=$EXTRA_TL
            export BATH_EPOCHS=80
            export BATH_SEED_OFFSET=$SEED
            export SLURM_ARRAY_TASK_ID=$GIDX
            taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
            echo "[multiseed extra] TL=$EXTRA_TL seed=$SEED g_idx=$GIDX exit=$?" \
                | tee -a "$LOGDIR/driver.log"
        ) &
        PIDS+=($!)
        slot=$((slot + 1))
    done
done

echo "[multiseed extra] launched ${#PIDS[@]} workers using $((slot * CORES_PER_TASK)) cores" \
    | tee -a "$LOGDIR/driver.log"
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[multiseed extra] all done" | tee -a "$LOGDIR/driver.log"

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
