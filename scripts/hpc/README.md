# Ejecución en Khipu

Ejecutar los comandos desde la raíz del repositorio.

## Preparación inicial

```bash
git clone --branch Modeling/pipaber \
  https://github.com/Codenid/capstone-claims-triage.git
cd capstone-claims-triage
uv sync
```

DVC usa un archivo local ignorado por Git para las credenciales:

```bash
uv run dvc remote modify --local dagshub auth basic
uv run dvc remote modify --local dagshub user YOUR_DAGSHUB_USERNAME
uv run dvc remote modify --local dagshub password YOUR_DAGSHUB_TOKEN
uv run dvc pull data/processed/prepared.parquet
```

No escribir el token en un script versionado ni compartirlo en el chat.

## MLflow

Crear el archivo privado a partir de la plantilla:

```bash
cp .env.example .env
chmod 600 .env
```

Completar `.env` directamente en Khipu. Para cargar sus valores:

```bash
set -a
source .env
set +a
uv run python src/evaluation/smoke_mlflow.py
```

En una máquina con SQLite 3.31 o posterior puede probarse MLflow con una base
local:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  uv run python src/evaluation/smoke_mlflow.py
```

Khipu tiene SQLite 3.26. Para su prueba local se permite temporalmente el backend
de archivos; los experimentos reales usarán DagsHub:

```bash
MLFLOW_ALLOW_FILE_STORE=true \
MLFLOW_TRACKING_URI=file:./mlruns \
  uv run python src/evaluation/smoke_mlflow.py
```

## Trabajos SLURM

Los nodos de cómputo no tienen acceso a internet. Por eso el entorno se prepara
con `uv sync` en el nodo de acceso y los scripts usan `uv run --no-sync`.

Verificar el contrato de entrada en CPU:

```bash
sbatch scripts/hpc/m0_verify.slurm
```

Verificar acceso a la A100:

```bash
sbatch scripts/hpc/gpu_smoke.slurm
```

Consultar los trabajos:

```bash
squeue -u piero.palacios
```
