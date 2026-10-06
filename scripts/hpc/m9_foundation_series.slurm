#!/usr/bin/env bash
#SBATCH --job-name=claims-m9-foundation
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export OMP_NUM_THREADS=8
export PYTENSOR_FLAGS=cxx=
export HF_HUB_OFFLINE=1

# Block C (models_plan.md §25.5): chronos2 or timesfm3, plus --smoke if wanted.
uv run --no-sync python -m src.models.weekly_counts.foundation_series "$@"
