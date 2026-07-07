#!/bin/bash
# Multi-seed sweep for TL=3 and TL=5 at g in {0.05, 0.10, 0.15, 0.20} in
# debugjob, to mirror what was done for TL=4 and enable apples-to-apples
# error-bar comparison.
#   debugjob 1 ./job_multiseed_TL35_debug.sh
#
# Layout: 40 workers (5 seeds × 4 g × 2 depths) at 4 cores/worker = 160 cores
# used. Each worker runs BATH_EPOCHS=80 (=96 total). Per-worker cost:
#   TL=3 ~600 s wall (2400 cpu-sec) — fits 5400 cap
#   TL=5 ~1300 s wall (5200 cpu-sec) — fits with slim margin
# Total wallclock dominated by TL=5: ~25 min, well within debugjob 1-h cap.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=4
LOGDIR=slurm_outputs/multiseed_TL35_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[multiseed TL3+TL5] logs in $LOGDIR" | tee "$LOGDIR/driver.log"

G_INDICES=(1 2 3 4)        # g = 0.05, 0.10, 0.15, 0.20 in COUPLINGS
SEED_OFFSETS=(0 1 2 3 4)
T_L_VALUES=(3 5)

slot=0
PIDS=()
for TL in "${T_L_VALUES[@]}"; do
    for SEED in "${SEED_OFFSETS[@]}"; do
        for GIDX in "${G_INDICES[@]}"; do
            START=$((slot * CORES_PER_TASK))
            END=$((START + CORES_PER_TASK - 1))
            TASK_LOG=$LOGDIR/worker_TL${TL}_s${SEED}_g${GIDX}.log
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
                export BATH_SEED_OFFSET=$((SEED + 100))   # avoid clobbering canonical seed-0 files
                export SLURM_ARRAY_TASK_ID=$GIDX
                taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
                echo "[multiseed] TL=$TL seed=$SEED g_idx=$GIDX exit=$?" | tee -a "$LOGDIR/driver.log"
            ) &
            PIDS+=($!)
            echo "[multiseed]   launched TL=$TL seed=$SEED g_idx=$GIDX cores=$START-$END" \
                | tee -a "$LOGDIR/driver.log"
            slot=$((slot + 1))
        done
    done
done

echo "[multiseed] launched ${#PIDS[@]} workers, waiting..." | tee -a "$LOGDIR/driver.log"
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[multiseed] all done" | tee -a "$LOGDIR/driver.log"

echo "=== Summary ===" | tee -a "$LOGDIR/driver.log"
for TL in "${T_L_VALUES[@]}"; do
    for SEED in "${SEED_OFFSETS[@]}"; do
        for GIDX in "${G_INDICES[@]}"; do
            LOG=$LOGDIR/worker_TL${TL}_s${SEED}_g${GIDX}.log
            [ -f "$LOG" ] || continue
            if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then
                tag="✓ full"
            elif grep -q "post-train save" "$LOG" 2>/dev/null; then
                tag="△ post-train only"
            else
                tag="✗ no save"
            fi
            echo "  TL=$TL seed=$SEED g_idx=$GIDX: $tag" | tee -a "$LOGDIR/driver.log"
        done
    done
done
