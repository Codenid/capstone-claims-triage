#!/usr/bin/env bash
#SBATCH --job-name=claims-final-confirmation
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# Pass --rehearsal first: it runs on calibration, checks the published results
# and saves nothing. Without it, this opens validation once.
uv run --no-sync python -m src.evaluation.final_confirmation "$@"
