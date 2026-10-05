#!/usr/bin/env bash
#SBATCH --job-name=claims-m5f
#SBATCH --partition=gpu
#SBATCH --qos=a-postgrado
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8
# The nodes have no internet: weights come from the files fetched on the login
# node, and nothing reports usage.
export HF_HUB_OFFLINE=1
export TABPFN_DISABLE_TELEMETRY=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --no-sync python -m src.models.foundation_t1 "$@"
