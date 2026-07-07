#!/bin/bash
# Test the packed-execution scheme inside a single 192-core debugjob:
#   debugjob 1 ./job_test_packed.sh
# (or `debugjob -n 192` — needs the full node).
#
# Launches 6 parallel Python workers, each pinned to one 32-core NUMA chiplet,
# each running one TL=3 task at max_bd=64. Predicted ~15 min per task; all 6
# should finish concurrently in ~15-20 min wallclock, far inside the 1 h
# debugjob cap.
set -uo pipefail
# debugjob inherits the login shell's RLIMIT_CPU = 3600 s, which kills any
# python process whose cpu-seconds (= cores × wall) exceed 1 h. Lift it
# explicitly so the lift propagates to all child subshells we spawn below.
ulimit -t unlimited || true
ulimit -v unlimited || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=32
N_PARALLEL=6
LOGDIR=slurm_outputs/test_packed_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[test_packed] logs in $LOGDIR" | tee "$LOGDIR/driver.log"

# Pin task i to cores [i*32 .. (i+1)*32-1]. Each worker also gets a unique
# SLURM_ARRAY_TASK_ID so run_bath_sweep.py picks the right (T_L, g) slot.
# All six are TL=3 tasks (g_idx 0..5), via BATH_TL_FILTER=3.
PIDS=()
for offset in $(seq 0 $((N_PARALLEL-1))); do
    START=$((offset * CORES_PER_TASK))
    END=$((START + CORES_PER_TASK - 1))
    TASK_LOG=$LOGDIR/worker_${offset}_g$((offset*5)).log
    echo "[test_packed] worker offset=$offset cores=$START-$END SLURM_ARRAY_TASK_ID=$offset → $TASK_LOG" \
        | tee -a "$LOGDIR/driver.log"
    (
        export OMP_NUM_THREADS=$CORES_PER_TASK
        export MKL_NUM_THREADS=$CORES_PER_TASK
        export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
        export GOMP_CPU_AFFINITY="$START-$END"
        export OMP_PROC_BIND=close
        export OMP_PLACES="{$START}:$CORES_PER_TASK"
        export BATH_N=10
        export BATH_TL_FILTER=3
        export BATH_EPOCHS=${BATH_EPOCHS:-12}  # small for debugjob CPU rlimit; override for production
        export SLURM_ARRAY_TASK_ID=$offset
        # taskset confines the kernel scheduler to this chiplet too, on top of OpenMP pinning.
        taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
        echo "[test_packed] worker offset=$offset exit=$?" | tee -a "$LOGDIR/driver.log"
    ) &
    PIDS+=($!)
done

echo "[test_packed] launched ${#PIDS[@]} workers, waiting…" | tee -a "$LOGDIR/driver.log"
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "[test_packed] all workers done" | tee -a "$LOGDIR/driver.log"

echo
echo "=== Summary ===" | tee -a "$LOGDIR/driver.log"
for offset in $(seq 0 $((N_PARALLEL-1))); do
    LOG=$LOGDIR/worker_${offset}_g$((offset*5)).log
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then
        echo "  worker $offset: ✓ saved" | tee -a "$LOGDIR/driver.log"
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then
        echo "  worker $offset: △ post-train save only (testing block didn't finish)" \
            | tee -a "$LOGDIR/driver.log"
    else
        echo "  worker $offset: ✗ no save" | tee -a "$LOGDIR/driver.log"
    fi
    grep -E "task .*: finished in|finished in.*s ===" "$LOG" 2>/dev/null | tee -a "$LOGDIR/driver.log"
done
