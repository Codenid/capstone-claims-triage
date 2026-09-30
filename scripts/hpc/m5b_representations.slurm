#!/usr/bin/env bash
#SBATCH --job-name=claims-m5b-representations
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# Pass --smoke to check the whole comparison on a small sample first.
uv run --no-sync python -m src.models.representation_comparison "$@"
