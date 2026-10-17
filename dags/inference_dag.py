"""
### Health inference
Triggered by the **trained_models** asset, so predictions are refreshed whenever a new best model is selected.

Generates the synthetic test records, then cleans, encodes and scores them with the best model named
in `config.yml`. Replace `generate_synthetic_data` with an extract of real new records in production.
"""

from datetime import datetime

from airflow.sdk import dag, task

from health_disease_prediction.orchestration import (
    DEFAULT_ARGS,
    PREDICTIONS,
    TAGS,
    TRAINED_MODELS,
)


@dag(
    dag_id="health_inference",
    schedule=[TRAINED_MODELS],
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=TAGS,
    doc_md=__doc__,
)
def health_inference():
    # Project imports stay inside tasks so DAG parsing stays fast and does not need the ML libraries

    @task
    def generate_synthetic_data() -> str:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.inference import generate_synthetic_data

        return str(generate_synthetic_data(load_config()))

    @task(outlets=[PREDICTIONS])
    def predict(input_path: str) -> str:
        from pathlib import Path

        from health_disease_prediction.config import load_config
        from health_disease_prediction.inference import run_inference

        return str(run_inference(load_config(), Path(input_path)))

    predict(generate_synthetic_data())


health_inference()
