#!/usr/bin/env bash
#SBATCH --job-name=claims-m11d-daily-change
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# D-11 (models_plan.md §28.3) on the frozen daily winner: fit + calibration. Pass --validation only after
# that decision is recorded: it reports the 2025-H1 alerts once.
uv run --no-sync python -m src.models.daily_change "$@"
