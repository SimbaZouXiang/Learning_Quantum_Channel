#!/bin/bash
#SBATCH --job-name=pqc-large-t
#SBATCH --output=slurm-pqc-large-t-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --array=0-3
# Array index 0..3 → t = 3.0, 4.0, 5.0, 6.0.
# Each element runs one Learning_TDME_using_data_{30|40|50|60}.py on a full
# Trillium node. Inside, the script forks 10 workers (one per gamma) so all
# 10 grid points for that t train in parallel with ~19 threads each.

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian

module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate

pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1

export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"

# Map array index → t-value script
case "$SLURM_ARRAY_TASK_ID" in
    0) SCRIPT=Learning_TDME_using_data_30.py ;;
    1) SCRIPT=Learning_TDME_using_data_40.py ;;
    2) SCRIPT=Learning_TDME_using_data_50.py ;;
    3) SCRIPT=Learning_TDME_using_data_60.py ;;
    *) echo "Unknown task id $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "Array task $SLURM_ARRAY_TASK_ID -> $SCRIPT on $SLURMD_NODENAME starting $(date)"
python -u "$SCRIPT"
echo "Finished $(date)"
