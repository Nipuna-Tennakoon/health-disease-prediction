"""Shared Airflow assets and defaults for the DAGs in dags/.

Only the DAG files import this module, so the rest of the package does not depend on Airflow.

Assets chain the DAGs without hard-coded triggers: preprocessing publishes CLEANED_DATASET, which
schedules training; training publishes TRAINED_MODELS, which schedules inference.
"""

from datetime import timedelta

from airflow.sdk import Asset

CLEANED_DATASET = Asset("health_disease_prediction.cleaned_dataset")
TRAINED_MODELS = Asset("health_disease_prediction.trained_models")
PREDICTIONS = Asset("health_disease_prediction.predictions")

DEFAULT_ARGS = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

TAGS = ["health-disease-prediction"]
