#!/usr/bin/env bash
#SBATCH --job-name=claims-m5
#SBATCH --partition=gpu
#SBATCH --qos=a-postgrado
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --no-sync python src/models/bge_sample.py
