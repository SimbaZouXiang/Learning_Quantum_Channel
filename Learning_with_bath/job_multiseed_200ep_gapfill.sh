#!/bin/bash
# Gap-fill multi-seed run at BATH_EPOCHS=200 for TL=2 and TL=3 at
# g in {0.05, 0.10, 0.15, 0.20}. Target: 40 saved seeds per (T_L, g).
# Reads existing _e200 *.testing_loss.npy files to figure out which seeds are
# missing, then runs only those — using seed offsets 240..279 (a fresh range
# beyond anything currently on disk).
#   debugjob 1 ./job_multiseed_200ep_gapfill.sh
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

CORES_PER_TASK=2
BATCH_CAP=96         # 192 / 2 cores
TARGET_SEEDS=40
LOGDIR=slurm_outputs/multiseed_200ep_gapfill_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[gapfill] logs $LOGDIR  cores/worker=$CORES_PER_TASK  target=$TARGET_SEEDS per point" \
    | tee "$LOGDIR/driver.log"

# Build the missing-task list (TL, g_idx, seed_offset).
# Use offsets 240..(240+N-1) where N is the number missing at that point.
TASKS=()
for TL in 2 3; do
    if [ "$TL" = "2" ]; then BD=256; else BD=64; fi
    for G_IDX in 1 2 3 4; do
        case $G_IDX in 1) GTAG=005;; 2) GTAG=010;; 3) GTAG=015;; 4) GTAG=020;; esac
        have=$(ls /scratch/simba/Learning_with_bath/npy_outputs/ 2>/dev/null \
            | grep -c "T${TL}_L${TL}_g${GTAG}_p010_bd${BD}_s2.._e200_testing_loss.npy")
        need=$((TARGET_SEEDS - have))
        if [ $need -lt 0 ]; then need=0; fi
        echo "  TL=$TL g=$GTAG : have $have, need $need more" | tee -a "$LOGDIR/driver.log"
        for ((k=0; k<need; k++)); do
            SEED=$((240 + k))
            TASKS+=("$TL $G_IDX $SEED")
        done
    done
done
TOTAL=${#TASKS[@]}
echo "[gapfill] total missing tasks to run: $TOTAL" | tee -a "$LOGDIR/driver.log"

i=0
batch_num=0
while [ $i -lt $TOTAL ]; do
    batch_num=$((batch_num + 1))
    echo "[gapfill] -- batch $batch_num starting at task index $i --" | tee -a "$LOGDIR/driver.log"
    PIDS=()
    slot=0
    while [ $i -lt $TOTAL ] && [ $slot -lt $BATCH_CAP ]; do
        IFS=' ' read -r TL G_IDX SEED <<< "${TASKS[$i]}"
        START=$((slot * CORES_PER_TASK))
        END=$((START + CORES_PER_TASK - 1))
        TASK_LOG=$LOGDIR/worker_TL${TL}_s${SEED}_g${G_IDX}.log
        (
            export OMP_NUM_THREADS=$CORES_PER_TASK
            export MKL_NUM_THREADS=$CORES_PER_TASK
            export NUMEXPR_MAX_THREADS=$CORES_PER_TASK
            export GOMP_CPU_AFFINITY="$START-$END"
            export OMP_PROC_BIND=close
            export OMP_PLACES="{$START}:$CORES_PER_TASK"
            export BATH_N=10
            export BATH_TL_FILTER=$TL
            export BATH_EPOCHS=200
            export BATH_SEED_OFFSET=$SEED
            export BATH_FILE_EXTRA_SUFFIX=_e200
            export SLURM_ARRAY_TASK_ID=$G_IDX
            taskset -c "$START-$END" python -u run_bath_sweep.py > "$TASK_LOG" 2>&1
        ) &
        PIDS+=($!)
        slot=$((slot + 1))
        i=$((i + 1))
    done
    echo "[gapfill] batch $batch_num: ${#PIDS[@]} workers, waiting..." | tee -a "$LOGDIR/driver.log"
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[gapfill] batch $batch_num done" | tee -a "$LOGDIR/driver.log"
done

ok=0; partial=0; bad=0
for LOG in "$LOGDIR"/worker_*.log; do
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then ok=$((ok+1))
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then partial=$((partial+1))
    else bad=$((bad+1)); fi
done
echo "[gapfill] ✓ full save: $ok  △ post-train only: $partial  ✗ no save: $bad" \
    | tee -a "$LOGDIR/driver.log"
