#!/usr/bin/env bash
#SBATCH --job-name=claims-m4
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
uv run --no-sync python src/models/tfidf_models.py
