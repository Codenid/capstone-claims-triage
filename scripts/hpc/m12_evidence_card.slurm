#!/usr/bin/env bash
#SBATCH --job-name=claims-m12-card
#SBATCH --partition=gpu
#SBATCH --qos=a-postgrado
#SBATCH --gres=shard:a100_3g.20gb:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export OMP_NUM_THREADS=8
export PYTENSOR_FLAGS=cxx=
export HF_HUB_OFFLINE=1
export TABPFN_DISABLE_TELEMETRY=1

# Facts of the evidence card; pass --as-of YYYY-MM-DD for the deadline.
uv run --no-sync python -m src.evidence.facts "$@"
