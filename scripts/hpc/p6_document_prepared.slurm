#!/usr/bin/env bash
#SBATCH --job-name=claims-p6-document
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export OMP_NUM_THREADS=4

# The stage reads prepared.parquet one column at a time and writes the data
# dictionary and the cleaning report; dvc.lock records the new stage only.
# --single-item: the raw and interim tables are not on this clone, and the
# stage only needs the prepared table that finalize_prepared published.
uv run --no-sync dvc repro --single-item document_prepared
git status --short dvc.lock docs/diccionario reports/preparation
