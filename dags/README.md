# Airflow DAGs

Three DAGs, one per pipeline operation. Each task calls a function from the `health_disease_prediction` package, so the DAGs only orchestrate; all logic lives in `src/`.

| DAG | Trigger | Tasks |
|---|---|---|
| `health_preprocessing` | Manual | `download_raw_data` → `clean_dataset` → `validate_cleaned_data` |
| `health_training` | `cleaned_dataset` asset | `split_dataset` → `fit_preprocessors` → `train_model` (one mapped task per model, in parallel) → `package_models` → `select_best_model` |
| `health_inference` | `trained_models` asset | `generate_synthetic_data` → `predict` |

Assets chain the DAGs: a successful preprocessing run triggers training, and a new best model triggers inference.

## Setup (Airflow 3.1, local)

```bash
# 1. Airflow in its own environment, installed with the official constraints
uv venv .airflow-venv -p 3.10
VIRTUAL_ENV=.airflow-venv uv pip install "apache-airflow==3.1.0" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.0/constraints-3.10.txt"

# 2. This project into the same environment (without the constraints: they pin older ML libraries)
VIRTUAL_ENV=.airflow-venv uv pip install -e .

# 3. Point Airflow at this repository
export AIRFLOW_HOME=$PWD/.airflow
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export HDP_PROJECT_ROOT=$PWD

.airflow-venv/bin/airflow standalone   # UI on http://localhost:8080
```

Unpause the three DAGs and trigger `health_preprocessing`; training and inference follow automatically.

To run one DAG end to end without the scheduler:

```bash
.airflow-venv/bin/airflow dags test health_training
```

## Notes

- `HDP_PROJECT_ROOT` tells the package where `config.yml`, `data/` and `artifacts/` live. Set it on every Airflow worker.
- W&B credentials come from `.env` in the project root (`WANDB_API_KEY`, `WANDB_ENTITY`). Set `wandb.mode: disabled` in `config.yml` to run without W&B.
- `health_training` writes the selected best model back to `config.yml`, which `health_inference` reads.
- The same stages run without Airflow: `uv run hdp preprocess | train | predict | all`.
