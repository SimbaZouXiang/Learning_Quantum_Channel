#!/bin/bash
#SBATCH --job-name=peak-debug
#SBATCH --output=slurm-peak-debug-%j.out
#SBATCH --partition=debug
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
# ---------------------------------------------------------------------
# Quick-turnaround priority subset of the peak-location sweep (70 points,
# T=3 only), sized for the 1 h debug wall limit on a full node.  The full
# grid runs in job_paper_extras.sh on the normal queue; completed points
# are shared via Result_peak_sweep (both drivers skip existing outputs).
# ---------------------------------------------------------------------
set -uo pipefail
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
cd "/scratch/simba/Quantum Circuit Learning"

module load python/3.12.4 2>/dev/null || true
source /home/simba/.virtualenvs/QIP/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"

python -u run_peak_sweep.py --subset debug --workers 24 --threads-per-worker 8 \
    --outdir Result_peak_sweep
echo "Finished: $(date)"
