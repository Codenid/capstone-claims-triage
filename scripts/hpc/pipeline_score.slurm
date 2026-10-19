#!/usr/bin/env bash
#SBATCH --job-name=claims-pipeline-score
#SBATCH --partition=gpu
#SBATCH --qos=a-postgrado
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export HF_HUB_OFFLINE=1
export TABPFN_DISABLE_TELEMETRY=1
export UV_NO_SYNC=1

# Score a new table with the frozen components (models_plan.md §29).
#   INPUT=tabla.parquet OUTPUT=reports/pipeline/<nombre> sbatch scripts/hpc/pipeline_score.slurm
INPUT="${INPUT:?set INPUT=ruta/tabla.parquet}"
OUTPUT="${OUTPUT:?set OUTPUT=carpeta de salida}"
DEVICE="${DEVICE:-cuda}"

uv run --no-sync python -m src.pipeline score --input "${INPUT}" --output "${OUTPUT}" --device "${DEVICE}" "$@"
