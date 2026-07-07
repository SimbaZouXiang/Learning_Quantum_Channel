#!/bin/bash
#SBATCH --job-name=bath-TL2-pk
#SBATCH --output=slurm_outputs/slurm-bath-TL2-pk-%j.out
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
# T_L=2, max_bd=256, use_compressed=True, packed: 6 workers/node × 32 cores each.
# 11 g values in 2 sequential batches of 6. Per-task wallclock estimate ~10-15 min
# at 32-core chiplet (TL=2 is shallow; bd=256 adds modest QR cost in compress).
# 3 h cap leaves ample margin.
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
LOGDIR=slurm_outputs/packed_TL2_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
echo "[TL2 packed] job ${SLURM_JOB_ID}, logs in $LOGDIR" | tee "$LOGDIR/driver.log"

for BATCH_START in 0 6; do
    echo "[TL2 packed] batch starting at g_idx=$BATCH_START" | tee -a "$LOGDIR/driver.log"
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
            export SLURM_ARRAY_TASK_ID=$IDX
            taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
            echo "[TL2 packed] worker g_idx=$IDX exit=$?" | tee -a "$LOGDIR/driver.log"
        ) &
        PIDS+=($!)
        echo "[TL2 packed]   launched g_idx=$IDX on cores $START-$END" | tee -a "$LOGDIR/driver.log"
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[TL2 packed] batch done" | tee -a "$LOGDIR/driver.log"
done

echo "=== Summary ===" | tee -a "$LOGDIR/driver.log"
for IDX in $(seq 0 $((N_G-1))); do
    LOG=$LOGDIR/worker_g${IDX}.log
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then
        echo "  g_idx=$IDX: ✓ full save" | tee -a "$LOGDIR/driver.log"
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then
        echo "  g_idx=$IDX: △ post-train save only" | tee -a "$LOGDIR/driver.log"
    else
        echo "  g_idx=$IDX: ✗ no save" | tee -a "$LOGDIR/driver.log"
    fi
done
