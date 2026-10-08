#!/usr/bin/env bash
#SBATCH --job-name=claims-m8c-catalog
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export PYTENSOR_FLAGS=cxx=

# Reads the frozen assignments, the weekly counts, the current M9 predictions
# and the M11 alerts; picks three public narratives per pattern from the
# prepared table; draws the 40 series. Nothing is refitted.
uv run --no-sync python -m src.models.pattern_catalog
