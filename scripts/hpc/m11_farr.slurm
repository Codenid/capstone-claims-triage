#!/usr/bin/env bash
#SBATCH --job-name=claims-m11-farr
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# C-M11-FARR v1 (models_plan.md §31.2): design on fit + calibration; pass
# --validation only after the decision is recorded: 2025-H1 alerts once.
uv run --no-sync python -m src.models.persistent_change_farr "$@"
