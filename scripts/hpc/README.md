# Ejecución en Khipu

Ejecutar los comandos desde la raíz del repositorio.

## Preparación inicial

```bash
git clone --branch Modeling/pipaber \
  https://github.com/Codenid/capstone-claims-triage.git
cd capstone-claims-triage
uv sync
```

DVC usa `.dvc/config.local`, un archivo ignorado por Git. En la instalación
actual ya contiene las credenciales. El usuario de DagsHub es `pipaber`; no debe
confundirse con la cuenta `piero.palacios` usada para entrar a Khipu.

En una instalación nueva se configura así:

```bash
uv run dvc remote modify --local dagshub auth basic
uv run dvc remote modify --local dagshub user pipaber
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

Verificar el contrato reproducible de M1:

```bash
sbatch scripts/hpc/m1_experiment.slurm
```

El trabajo crea el registro sin conectarse a internet. Cuando termine,
publicarlo desde el nodo de acceso:

```bash
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/runs/m1-experiment-setup-khipu/run.json
```

Auditar y congelar la evaluación de M2:

```bash
sbatch scripts/hpc/m2_evaluation.slurm
```

Cuando termine, publicar el registro desde el nodo de acceso:

```bash
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/runs/m2-evaluation-contract/run.json
```

Ejecutar las referencias simples de M3:

```bash
sbatch scripts/hpc/m3_baselines.slurm
```

Publicar sus ocho runs desde el nodo de acceso:

```bash
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/runs/m3
```

Entrenar TF-IDF en CPU para M4:

```bash
sbatch scripts/hpc/m4_tfidf.slurm
```

Cuando termine, guardar los modelos grandes con DVC desde el nodo de acceso:

```bash
uv run --no-sync dvc add artifacts/models/tfidf
uv run --no-sync dvc push artifacts/models/tfidf.dvc
```

Después publicar los cuatro runs pequeños en MLflow:

```bash
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/runs/m4
```

Preparar BGE en el nodo de acceso:

```bash
uv sync --group gpu
uv run --no-sync python -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('BAAI/bge-large-en-v1.5'))"
```

El segundo comando descarga el modelo antes de entrar a SLURM, porque los nodos
de cómputo no tienen internet. Después se ejecuta la comparación M5:

```bash
sbatch scripts/hpc/m5_bge.slurm
```

M6 reutiliza los embeddings de M5 y se ejecuta en CPU. Preparar el entorno y
verificar las pruebas desde el nodo de acceso:

```bash
uv sync --group semantic
uv run --no-sync dvc pull artifacts/models/bge_sample.dvc
uv run --no-sync python -m unittest tests.test_semantic_space -v
```

Enviar PCA, UMAP y FAISS a SLURM:

```bash
sbatch scripts/hpc/m6_semantic_space.slurm
```

Cuando termine, guardar el artefacto grande con DVC:

```bash
uv run --no-sync dvc add artifacts/models/semantic_space
uv run --no-sync dvc push artifacts/models/semantic_space.dvc
```

Después publicar el run pequeño en MLflow:

```bash
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/runs/m6/semantic_space/run.json
```

M7 compara clustering sobre una muestra común del periodo de ajuste:

```bash
uv sync --group semantic
uv run --no-sync dvc pull artifacts/models/bge_sample.dvc
uv run --no-sync dvc pull artifacts/models/semantic_space.dvc
uv run --no-sync python -m unittest tests.test_cluster_comparison -v
sbatch scripts/hpc/m7_cluster_comparison.slurm
```

Al terminar, guardar el artefacto y publicar el run agregado:

```bash
uv run --no-sync dvc add artifacts/models/clustering_comparison
uv run --no-sync dvc push artifacts/models/clustering_comparison.dvc
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/runs/m7/cluster_comparison/run.json
```

M8A genera embeddings para todos los reclamos elegibles y reutiliza los de M5:

```bash
uv sync --group gpu --group semantic
uv run --no-sync dvc pull data/processed/prepared.parquet
uv run --no-sync dvc pull artifacts/models/bge_sample.dvc
uv run --no-sync python -m unittest tests.test_bge_full -v
sbatch scripts/hpc/m8a_bge_full.slurm
```

El job guarda checkpoints en `artifacts/models/.bge_full.inprogress`. Si SLURM lo
interrumpe, se envía nuevamente el mismo script y continúa desde el último
bloque confirmado. No borrar ese directorio durante una ejecución incompleta.
Antes de finalizar, divide los arreglos que superen 800,000 filas en partes de
hasta 200,000 filas para respetar los límites de tamaño y tiempo del remoto DVC.

Al terminar:

```bash
uv run --no-sync dvc add artifacts/models/bge_full
uv run --no-sync dvc push artifacts/models/bge_full.dvc
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/runs/m8a/bge_full/run.json
```

Antes de M8B, copiar el hash de `artifacts/models/bge_full.dvc` a
`weekly_patterns.source_dvc_hash` en `configs/modeling.yaml`. M8B transforma el
corpus completo, ajusta `k=40` solo con `fit` y genera conteos semanales:

```bash
uv sync --group semantic
uv run --no-sync dvc pull artifacts/models/bge_full.dvc
uv run --no-sync dvc pull artifacts/models/semantic_space.dvc
uv run --no-sync dvc pull artifacts/models/clustering_comparison.dvc
uv run --no-sync python -m unittest tests.test_weekly_patterns -v
sbatch scripts/hpc/m8b_weekly_patterns.slurm
```

El job guarda cada bloque UMAP en
`artifacts/models/.weekly_patterns.inprogress`. Si se interrumpe, enviar el mismo
script otra vez sin borrar ese directorio. Al terminar:

```bash
uv run --no-sync dvc add artifacts/models/weekly_patterns
uv run --no-sync dvc push artifacts/models/weekly_patterns.dvc
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/runs/m8b/weekly_patterns/run.json
```

Si la subida conjunta de M8 supera la duración de la conexión SSH, puede dejarse
ejecutando en el nodo de acceso:

```bash
nohup uv run --no-sync dvc push --jobs 1 \
  artifacts/models/bge_full.dvc \
  artifacts/models/weekly_patterns.dvc \
  > reports/modeling/dvc-push-m8.log 2>&1 < /dev/null &
```

Después de que termine, comprobar que ambos artefactos existen en el remoto:

```bash
uv run --no-sync dvc status -c artifacts/models/bge_full.dvc
uv run --no-sync dvc status -c artifacts/models/weekly_patterns.dvc
```

Ambos comandos deben indicar que los datos y la caché están actualizados.

M5B compara las representaciones de T1–T4 entrenadas con todo `fit`
(`models_plan.md` §21). Usa los embeddings de M8A y los modelos de M3, M4 y M5,
así que corre en CPU. Primero se lanza una prueba rápida con una muestra, que no
guarda nada, y después el run completo:

```bash
uv run --no-sync python -m unittest tests.test_representation_comparison -v
sbatch scripts/hpc/m5b_representations.slurm --smoke
sbatch scripts/hpc/m5b_representations.slurm
```

El run completo se detiene antes de entrenar si M3 o M4 no reproducen sus
métricas publicadas, y nunca sobrescribe resultados anteriores. Al terminar:

```bash
uv run --no-sync dvc add artifacts/models/representation_comparison
uv run --no-sync dvc push artifacts/models/representation_comparison.dvc
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py reports/modeling/runs/m5b
```

M9 ajusta el modelo jerárquico Negative Binomial solo con las semanas completas
de `fit`. Calibración y validación se predicen sin actualizar el posterior:

```bash
uv sync --group probabilistic
uv run --no-sync python -m unittest \
  tests.test_negative_binomial \
  tests.test_weekly_count_models -v
sbatch scripts/hpc/m9_negative_binomial.slurm
```

El job ejecuta el entry point modular
`python -m src.models.weekly_counts.run` con la configuración
`configs/weekly_counts/nb_softmax_linear_v2.yaml`. Usa PyMC con cuatro cadenas
NUTS de NumPyro/JAX en CPU. Además desactiva el compilador C de PyTensor para
crear los puntos iniciales, ya que los nodos SLURM no tienen los encabezados de
desarrollo del sistema.

Cada ejecución recibe un `run_key` único y escribe primero en directorios
`.inprogress` dentro de `reports/modeling/weekly_counts/<model_id>/` y
`artifacts/models/weekly_counts/<model_id>/`. Si se interrumpe, conservar esos
directorios para revisar el posterior parcial antes de decidir si se repite. El
log de SLURM muestra el `run_key` y las rutas finales cuando el job termina.
Después, versionar la colección de artefactos y publicar el registro específico:

```bash
uv run --no-sync dvc add artifacts/models/weekly_counts
uv run --no-sync dvc push artifacts/models/weekly_counts.dvc
set -a
source .env
set +a
uv run --no-sync python src/evaluation/publish_run.py \
  reports/modeling/weekly_counts/nb_softmax_linear_v2/RUN_KEY/run.json
```

MLflow recibirá la fórmula, los priors, la configuración del muestreo,
diagnósticos de convergencia, métricas de backtest, predicciones y gráficos. El
run conserva un `candidate_status` de rechazo cuando no pasa todos los
criterios; un job terminado correctamente no implica que el modelo haya sido
aceptado.

Los modelos semanales nuevos usan un único script genérico. Recibe la
configuración en `MODEL_CONFIG` y el nivel de ejecución en `RUN_MODE`
(`prior`, `pilot` o `full`; por defecto `full`):

```bash
sbatch --export=ALL,MODEL_CONFIG=configs/weekly_counts/poisson_static_pymc_v1.yaml,RUN_MODE=full \
  scripts/hpc/m9_weekly_count.slurm
```

- `prior` solo genera el prior predictive con las semanas de `fit` y escribe
  `reports/modeling/weekly_counts/<model_id>/prior_checks/<run_key>.json`. Tarda
  segundos, por lo que también puede ejecutarse en el nodo de acceso con
  `PYTENSOR_FLAGS=cxx= uv run --no-sync python -m src.models.weekly_counts.run
  --config CONFIG --run-mode prior`.
- `pilot` usa 2 cadenas, 250 de tune y 250 draws sin modificar la configuración
  congelada. Su estado es siempre `candidate_status=pilot_only`; sirve para
  detectar problemas técnicos, no para elegir modelos. Queda solo en Git: no
  se publica en MLflow.
- `full` usa exactamente el muestreo de la configuración.

`publish_run.py` solo acepta runs `full` y rechaza un `run_name` que ya exista
en MLflow. Así cada run registra una sola decisión, sin duplicados.

El nivel queda en el `run_key`, en `results.json` y en el tag `run_mode` de
MLflow. `candidate_role` se lee de `configs/weekly_counts/releases.json`; B1
PyMC figura como `pipeline_baseline`: es una referencia del pipeline, no un
candidato a promoción.

Cada run también calcula los baselines móviles B1-R4 y B1-R13 y compara el
modelo con el mejor baseline mediante un bootstrap semanal pareado; el
resultado queda en `bootstrap.json`. Las configuraciones que declaran
`acceptance.rule: best_baseline_bootstrap_v1` usan la regla de promoción de
`models_plan.md` (§12.4); las congeladas antes conservan su regla. Los runs no
evalúan ni grafican validación: sus predicciones quedan en `predictions.csv`
para la confirmación final única.

Los baselines B1-R4 y B1-R13 se registran como runs propios. No usan MCMC y
tardan segundos, así que se ejecutan en el nodo de acceso:

```bash
CLAIMS_EXECUTION_HOST=khipu uv run --no-sync python -m src.models.weekly_counts.baselines
```

El comando escribe `reports/modeling/weekly_counts/b1_rolling_4_v1/<run_key>/` y
`reports/modeling/weekly_counts/b1_rolling_13_v1/<run_key>/`, con métricas solo
de calibración. Cada `run.json` se publica con `publish_run.py`.

M10 modela la composición semanal conjunta con un runner propio,
`src.models.weekly_composition.run`, y un script SLURM equivalente. Los
baselines B2 y B2-R4 se calculan dentro de cada run, sin MCMC. El prior check se
ejecuta en el nodo de acceso; el piloto y el full, con SLURM:

```bash
CLAIMS_EXECUTION_HOST=khipu PYTENSOR_FLAGS=cxx= MPLBACKEND=Agg \
  uv run --no-sync python -m src.models.weekly_composition.run \
  --config configs/weekly_composition/dirichlet_multinomial_rolling_4_v1.yaml \
  --run-mode prior
sbatch --export=ALL,MODEL_CONFIG=configs/weekly_composition/dirichlet_multinomial_rolling_4_v1.yaml,RUN_MODE=pilot \
  scripts/hpc/m10_weekly_composition.slurm
```

Los resultados quedan en `reports/modeling/weekly_composition/<model_id>/` y
`artifacts/models/weekly_composition/<model_id>/`. Después de un full:

```bash
uv run --no-sync dvc add artifacts/models/weekly_composition
uv run --no-sync dvc push artifacts/models/weekly_composition.dvc
```

La confirmación final en validación (`models_plan.md` §22) se ejecuta una sola
vez. Antes se ensaya sobre calibración, donde debe reproducir los resultados ya
publicados sin guardar nada:

```bash
sbatch scripts/hpc/final_confirmation.slurm --rehearsal
sbatch scripts/hpc/final_confirmation.slurm
```

El segundo comando abre validación, escribe
`reports/modeling/final_confirmation.json` y el registro
`reports/modeling/runs/final_confirmation/run.json`. Se niega a correr si el
reporte ya existe.

M11 (`models_plan.md` §24) usa el posterior de NB-R4-H v3 que dejó M9 en
`artifacts/models/weekly_counts`. El primer comando trabaja solo con `fit` +
calibración: fija los umbrales, prueba los aumentos artificiales y elige la
regla. El segundo reporta una sola vez las alertas de validación y solo se
ejecuta después de registrar esa decisión:

```bash
sbatch scripts/hpc/m11_persistent_change.slurm
sbatch scripts/hpc/m11_persistent_change.slurm --validation
```

Los resultados quedan en `reports/modeling/persistent_change/` y los registros
en `reports/modeling/runs/persistent_change_<split>/run.json`. Ninguno de los
dos comandos sobrescribe resultados existentes.

La demo de M12 (`models_plan.md` §25) necesita los artefactos de M6 (índice
FAISS) y de M8 (patrones). Desde el nodo de acceso, que tiene internet:

```bash
uv run --no-sync dvc pull artifacts/models/semantic_space.dvc \
  artifacts/models/weekly_patterns.dvc
sbatch scripts/hpc/m12_triage_demo.slurm
```

Escribe el panel, las fichas y su índice en `reports/modeling/triage_demo/`, y
el registro en `reports/modeling/runs/triage_demo/run.json`. Se niega a correr
si la demo ya existe.

Verificar acceso a la A100:

```bash
sbatch scripts/hpc/gpu_smoke.slurm
```

Consultar los trabajos activos:

```bash
squeue -u piero.palacios
```

Cuando un trabajo ya no aparezca en `squeue`, revisar su resultado y consumo:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```
