#!/bin/bash
# Submits every configs/runs/*.yaml as one Slurm array task each:
#   sbatch scripts/submit_array.sbatch
# Update --array's range below to match the current number of files in
# configs/runs/ (17 for phases 0-2; ls configs/runs/*.yaml | wc -l tells you).
# Adding a 28th run costs a new YAML there and bumping this range by one --
# never a new .sbatch file.
#SBATCH --job-name=beamsurrogate_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=medium
#SBATCH --array=0-16
set -euo pipefail

REPO_ROOT="/home/aph25/Code_GH"     # <-- update to this repo's actual path on the cluster
cd "$REPO_ROOT"
mkdir -p logs

CONFIGS=(configs/runs/*.yaml)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Matches --cpus-per-task above (NOT 1) -- see run.job's comment: torch's
# intra-op matmul threading gives a real speedup at the full dataset's
# batch size, on a job that owns its whole Slurm allocation.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8
PYTHON=/home/aph25/Desktop/wave_env/bin/python

echo "Array task $SLURM_ARRAY_TASK_ID -> $CONFIG"
time "$PYTHON" -u -m beamsurrogate --config "$CONFIG"
echo "RUN DONE: $CONFIG"
