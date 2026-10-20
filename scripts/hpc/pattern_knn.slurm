#!/usr/bin/env bash
#SBATCH --job-name=claims-pattern-knn
#SBATCH --partition=standard
#SBATCH --qos=a-postgrado
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=reports/modeling/slurm-%x-%j.out

set -euo pipefail

export CLAIMS_EXECUTION_HOST=khipu
export MPLBACKEND=Agg
export PYTENSOR_FLAGS=cxx=
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# C-PAT-INC v1 (models_plan.md §31.1): pattern by the vote of the nearest fit
# complaints in the frozen FAISS index, compared with the frozen M8B labels.
uv run --no-sync python -m src.models.pattern_knn "$@"
