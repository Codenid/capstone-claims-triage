#!/usr/bin/env bash
#SBATCH --job-name=claims-pb24-checks
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# PB-24 checks without a model change (models_plan.md §32.6): the WIS curve
# over the discount of C-A, and the boundary mass of each pattern.
uv run --no-sync python -m src.models.weekly_counts.discount_curve
uv run --no-sync python -m src.models.pattern_margin
