#!/usr/bin/env bash
#SBATCH --job-name=claims-m10-signal
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export OMP_NUM_THREADS=4
export PYTENSOR_FLAGS=cxx=

# Posterior predictive p-value of M10's joint log score per week (§27).
# CPU only: it reads the frozen DM-R4 posterior and simulates compositions.
uv run --no-sync python -m src.models.weekly_composition.mixture_signal "$@"
