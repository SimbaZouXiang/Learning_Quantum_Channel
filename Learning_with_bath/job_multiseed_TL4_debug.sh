#!/bin/bash
# Multi-seed TL=4 sweep at g in {0.05, 0.10, 0.15, 0.20} in debugjob.
#   debugjob 1 ./job_multiseed_TL4_debug.sh
#
# Layout: 20 workers (5 seeds × 4 g) at 4 cores/worker = 80 cores used (out of
# 192). Each worker runs BATH_EPOCHS=80 (= 96 total epochs); estimated wall
# ~1000-1200 s per worker → ~4000-4800 cpu-sec → fits in 5400 cpu-sec rlimit.
# Total wall (all 20 in parallel): ~20 min, well inside 1 h debugjob cap.
#
# Files saved with `_s{offset}` suffix so they don't clobber the canonical
# seed-0 sweep.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=4
LOGDIR=slurm_outputs/multiseed_TL4_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[multiseed TL4] logs in $LOGDIR" | tee "$LOGDIR/driver.log"

# Map global slot offset → (seed_offset, g_idx). We sweep g_idx in {0,1,2,3}
# (which is COUPLINGS index for 0.05, 0.10, 0.15, 0.20 after dropping g=0).
# Actually COUPLINGS = [0.0, 0.05, 0.10, 0.15, 0.20, ...] so g_idx in {1,2,3,4}.
G_INDICES=(1 2 3 4)
SEED_OFFSETS=(0 1 2 3 4)

slot=0
PIDS=()
for SEED in "${SEED_OFFSETS[@]}"; do
    for GIDX in "${G_INDICES[@]}"; do
        START=$((slot * CORES_PER_TASK))
        END=$((START + CORES_PER_TASK - 1))
        TASK_LOG=$LOGDIR/worker_s${SEED}_g${GIDX}.log
        (
            export OMP_NUM_THREADS=$CORES_PER_TASK
            export MKL_NUM_THREADS=$CORES_PER_TASK
            export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
            export GOMP_CPU_AFFINITY="$START-$END"
            export OMP_PROC_BIND=close
            export OMP_PLACES="{$START}:$CORES_PER_TASK"
            export BATH_N=10
            export BATH_TL_FILTER=4
            export BATH_EPOCHS=80
            # Offset by +100 so even seed_offset=0 in this batch produces
            # `_s100`-suffixed files rather than clobbering the original
            # canonical (seed=0, BATH_EPOCHS=100) sweep on disk.
            export BATH_SEED_OFFSET=$((SEED + 100))
            export SLURM_ARRAY_TASK_ID=$GIDX
            taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
            echo "[multiseed] seed=$SEED g_idx=$GIDX exit=$?" | tee -a "$LOGDIR/driver.log"
        ) &
        PIDS+=($!)
        echo "[multiseed]   launched seed=$SEED g_idx=$GIDX cores=$START-$END" \
            | tee -a "$LOGDIR/driver.log"
        slot=$((slot + 1))
    done
done

echo "[multiseed] launched ${#PIDS[@]} workers, waiting..." | tee -a "$LOGDIR/driver.log"
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[multiseed] all done" | tee -a "$LOGDIR/driver.log"

echo "=== Summary ===" | tee -a "$LOGDIR/driver.log"
for SEED in "${SEED_OFFSETS[@]}"; do
    for GIDX in "${G_INDICES[@]}"; do
        LOG=$LOGDIR/worker_s${SEED}_g${GIDX}.log
        [ -f "$LOG" ] || continue
        if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then
            tag="✓ full"
        elif grep -q "post-train save" "$LOG" 2>/dev/null; then
            tag="△ post-train only"
        else
            tag="✗ no save"
        fi
        echo "  seed=$SEED g_idx=$GIDX: $tag" | tee -a "$LOGDIR/driver.log"
    done
done
