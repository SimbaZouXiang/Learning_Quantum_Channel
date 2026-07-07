#!/bin/bash
# Parameterised 200-epoch multi-seed driver for one debugjob.
# Env vars:
#   E200_TL              depth (2, 3, 4, 5)
#   E200_GIDX_LIST       space-separated g_idx list (in COUPLINGS, e.g. "1 2 3 4" for g=0.05..0.20)
#   E200_SEED_START      first seed offset (default 100, so files _s100_e200...)
#   E200_SEED_END        last seed offset inclusive (default 119 = 20 seeds)
#   E200_CORES           cores per worker (default 2)
#   E200_BATCH_CAP       max parallel workers per batch (default 96)
#
# Skips (TL,g,seed) pairs whose _e200 testing_loss.npy already exists, so the
# script is idempotent and can be re-run after a partial debugjob.
set -uo pipefail
ulimit -t unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
cd /scratch/simba/Learning_with_bath
mkdir -p slurm_outputs npy_outputs
source /home/simba/.virtualenvs/QIP/bin/activate
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR"

: "${E200_TL:?must set E200_TL}"
: "${E200_GIDX_LIST:?must set E200_GIDX_LIST}"
E200_SEED_START=${E200_SEED_START:-100}
E200_SEED_END=${E200_SEED_END:-119}
E200_CORES=${E200_CORES:-2}
E200_BATCH_CAP=${E200_BATCH_CAP:-96}

if [ "$E200_TL" = "2" ]; then BD=256; else BD=64; fi

LOGDIR=slurm_outputs/200ep_TL${E200_TL}_$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"
echo "[200ep TL=$E200_TL] g_idx={$E200_GIDX_LIST}  seeds $E200_SEED_START..$E200_SEED_END  cores/worker=$E200_CORES" \
    | tee "$LOGDIR/driver.log"

# Build missing-task list, skipping any already-saved (TL, g, seed) tuple.
TASKS=()
for SEED in $(seq "$E200_SEED_START" "$E200_SEED_END"); do
    for G_IDX in $E200_GIDX_LIST; do
        case $G_IDX in
            1) GTAG=005;; 2) GTAG=010;; 3) GTAG=015;; 4) GTAG=020;;
            5) GTAG=025;; 6) GTAG=030;; 7) GTAG=035;; 8) GTAG=040;;
            9) GTAG=045;; 10) GTAG=050;;
            *) echo "unknown g_idx=$G_IDX"; continue;;
        esac
        DONE_FILE="npy_outputs/bath_sweep_N10_T${E200_TL}_L${E200_TL}_g${GTAG}_p010_bd${BD}_s${SEED}_e200_testing_loss.npy"
        if [ -f "$DONE_FILE" ]; then
            continue
        fi
        TASKS+=("$G_IDX $SEED")
    done
done
TOTAL=${#TASKS[@]}
echo "[200ep TL=$E200_TL] missing tasks to run: $TOTAL  batch cap: $E200_BATCH_CAP" \
    | tee -a "$LOGDIR/driver.log"

i=0
batch_num=0
while [ $i -lt $TOTAL ]; do
    batch_num=$((batch_num + 1))
    echo "[200ep TL=$E200_TL] -- batch $batch_num starting at task index $i --" \
        | tee -a "$LOGDIR/driver.log"
    PIDS=()
    slot=0
    while [ $i -lt $TOTAL ] && [ $slot -lt $E200_BATCH_CAP ]; do
        IFS=' ' read -r G_IDX SEED <<< "${TASKS[$i]}"
        START=$((slot * E200_CORES))
        END=$((START + E200_CORES - 1))
        TASK_LOG=$LOGDIR/worker_s${SEED}_g${G_IDX}.log
        (
            export OMP_NUM_THREADS=$E200_CORES
            export MKL_NUM_THREADS=$E200_CORES
            export NUMEXPR_MAX_THREADS=$E200_CORES
            export GOMP_CPU_AFFINITY="$START-$END"
            export OMP_PROC_BIND=close
            export OMP_PLACES="{$START}:$E200_CORES"
            export BATH_N=10
            export BATH_TL_FILTER=$E200_TL
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
    echo "[200ep TL=$E200_TL] batch $batch_num: ${#PIDS[@]} workers, waiting..." \
        | tee -a "$LOGDIR/driver.log"
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "[200ep TL=$E200_TL] batch $batch_num done" | tee -a "$LOGDIR/driver.log"
done

ok=0; partial=0; bad=0
for LOG in "$LOGDIR"/worker_*.log; do
    [ -f "$LOG" ] || continue
    if grep -q "saved.*\.npy" "$LOG" 2>/dev/null; then ok=$((ok+1))
    elif grep -q "post-train save" "$LOG" 2>/dev/null; then partial=$((partial+1))
    else bad=$((bad+1)); fi
done
echo "[200ep TL=$E200_TL] ✓ full save: $ok  △ post-train only: $partial  ✗ no save: $bad" \
    | tee -a "$LOGDIR/driver.log"
