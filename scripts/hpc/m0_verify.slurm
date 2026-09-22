#!/usr/bin/env bash
#SBATCH --job-name=claims-m0
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

uv run --no-sync python src/evaluation/verify_modeling_input.py
uv run --no-sync dvc status -c data/processed/prepared.parquet
