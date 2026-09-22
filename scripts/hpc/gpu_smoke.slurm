#!/usr/bin/env bash
#SBATCH --job-name=claims-gpu-smoke
#SBATCH --partition=gpu
#SBATCH --qos=a-postgrado
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run python -c "import platform; print(platform.python_version())"
