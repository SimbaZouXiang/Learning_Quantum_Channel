#!/bin/bash
#SBATCH --job-name=paper-extras
#SBATCH --output=slurm-paper-extras-%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mail-user=xiang.zou@mail.utoronto.ca
#SBATCH --mail-type=END
# ---------------------------------------------------------------------
# ONE packed full-node job for all remaining paper simulations:
#   (1) peak-location sweep, full grid  (486 points, 24 workers x 8 threads)
#   (2) N=12 measured-LOE with truncation (d_max=256)
#   (3) N=30 truncation-error spot check (d_max 32/64/128 forward passes)
# Each section is resumable: rerunning skips completed points.
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

echo "=== [1/3] peak sweep (full grid) ==="
python -u run_peak_sweep.py --subset full --workers 24 --threads-per-worker 8 \
    --outdir Result_peak_sweep
echo "peak sweep exit: $?"

echo "=== [2/3] N=12 LOE (truncated, d_max=256) ==="
LOE_THREADS=48 OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 \
python -u measure_loe.py --N 12 --Lmax 12 --seeds 2 \
    --plist 0,0.01,0.02,0.03,0.05,0.1 --truncation --max-bd 256 \
    --out loe_measurement
echo "N=12 LOE exit: $?"

echo "=== [3/3] N=30 truncation spot check ==="
TRUNC_THREADS=96 OMP_NUM_THREADS=96 MKL_NUM_THREADS=96 \
python -u check_truncation_N30.py
echo "truncation check exit: $?"

echo "Finished: $(date)"
