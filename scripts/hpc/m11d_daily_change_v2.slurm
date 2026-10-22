#!/usr/bin/env bash
#SBATCH --job-name=claims-m11d-daily-change-v2
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# D-11 v2 (models_plan.md §32.5): 3-day scan and empirical thresholds on D-D; fit + calibration.
# Pass --validation only after the decision is recorded: 2025-H1 alerts once.
uv run --no-sync python -m src.models.daily_change_v2 "$@"
