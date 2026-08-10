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
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=short
#SBATCH --array=0-16
set -euo pipefail

REPO_ROOT="/home/aph25/Code_GH"     # <-- update to this repo's actual path on the cluster
cd "$REPO_ROOT"
mkdir -p logs

CONFIGS=(configs/runs/*.yaml)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PYTHON=/home/aph25/Desktop/wave_env/bin/python

echo "Array task $SLURM_ARRAY_TASK_ID -> $CONFIG"
time "$PYTHON" -u -m beamsurrogate --config "$CONFIG"
echo "RUN DONE: $CONFIG"
