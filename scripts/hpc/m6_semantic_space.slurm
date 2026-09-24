#!/usr/bin/env bash
#SBATCH --job-name=claims-m6
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMBA_NUM_THREADS=1

uv run --no-sync python src/models/semantic_space.py
