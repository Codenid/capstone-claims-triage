#!/usr/bin/env bash
#SBATCH --job-name=claims-m9-discount-grid
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export OMP_NUM_THREADS=2

# C-A (models_plan.md §25.5): chooses the discount and cap in fit only.
uv run --no-sync python -m src.models.weekly_counts.discount_grid
