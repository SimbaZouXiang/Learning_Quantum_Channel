#!/bin/bash
#SBATCH --job-name=tdme-validate
#SBATCH --output=slurm-tdme-validate-%j.out
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192

set -euo pipefail
cd /scratch/simba/Learning_Lindbladian/tdme_save_params
module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMBA_NUM_THREADS=64
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
echo "Start: $(date)  Node: $SLURMD_NODENAME"
python -u build_tdme_dense.py
echo "End:   $(date)"
