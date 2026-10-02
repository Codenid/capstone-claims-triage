#!/usr/bin/env bash
#SBATCH --job-name=claims-m12-brief-facts
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# The exports have no narratives and stay outside the repository.
EXPORT_DIR="${BRIEF_EXPORT_DIR:-$HOME/m12_brief}"

uv run --no-sync python -m src.triage.brief_facts --export "$EXPORT_DIR"
uv run --no-sync python -m src.triage.brief_facts \
  --rows "$EXPORT_DIR/complaint_rows.parquet" \
  --history "$EXPORT_DIR/pattern_history.parquet"
