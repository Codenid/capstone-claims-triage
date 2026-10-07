#!/usr/bin/env bash
#SBATCH --job-name=claims-p1-5-prep
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export OMP_NUM_THREADS=4

# Re-run the whole preparation chain after moving its decisions to params.yaml
# (PB-11). The raw parquet must be pulled first; every stage is re-executed
# because its deps and params changed. The tables must come out byte-identical:
# the MD5 of the prepared table is compared with the frozen contract below.
EXPECTED_MD5=d189a3ae6ea7a8f9041fc8210968eda6
uv run --no-sync dvc repro finalize_prepared
uv run --no-sync dvc repro --single-item document_prepared
ACTUAL_MD5=$(md5sum data/processed/prepared.parquet | cut -d' ' -f1)
echo "prepared.parquet md5: ${ACTUAL_MD5} (expected ${EXPECTED_MD5})"
if [ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]; then
  echo "MD5 CHANGED: do not commit dvc.lock; stop and review." >&2
  exit 1
fi
git status --short dvc.lock docs/diccionario reports/preparation
git diff --stat dvc.lock
