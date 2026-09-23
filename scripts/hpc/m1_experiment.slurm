#!/usr/bin/env bash
#SBATCH --job-name=claims-m1
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
uv run --no-sync python src/evaluation/verify_experiment_setup.py \
  --output reports/modeling/runs/m1-experiment-setup-khipu/run.json
