#!/usr/bin/env bash
#SBATCH --job-name=claims-m9-state-space
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

RUN_MODE="${RUN_MODE:-full}"

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTENSOR_FLAGS=cxx=
export XLA_FLAGS=--xla_force_host_platform_device_count=4
export JAX_ENABLE_X64=true

# C-B (models_plan.md §25.5): RUN_MODE=prior, pilot or full.
uv run --no-sync python -m src.models.weekly_counts.state_space \
  --config configs/weekly_counts/nb_state_space_v1.yaml \
  --run-mode "${RUN_MODE}"
