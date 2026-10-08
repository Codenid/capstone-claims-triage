#!/usr/bin/env bash
#SBATCH --job-name=claims-m9d-state-space
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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

uv run --no-sync python -m src.models.daily_counts.state_space --run-mode "${RUN_MODE}"
