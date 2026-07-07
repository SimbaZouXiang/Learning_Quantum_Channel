#!/bin/bash
#SBATCH --job-name=three-stu-an
#SBATCH --output=slurm-three-stu-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian/mpo_save_params
module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate

pkill -9 -u "$USER" -f "TDME_Trott" 2>/dev/null || true
sleep 1

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"

echo "Start: $(date)  Node: $SLURMD_NODENAME"
python -u analyze_three_student_distances.py
echo "End:   $(date)"
