"""
### Health preprocessing
Downloads the raw patient records (skipped if present), cleans them into a model-ready dataset,
saves the fill values inference needs, and validates the result.

A successful validation publishes the **cleaned_dataset** asset, which triggers the training DAG.
"""

from datetime import datetime

from airflow.sdk import dag, task

from health_disease_prediction.orchestration import CLEANED_DATASET, DEFAULT_ARGS, TAGS


@dag(
    dag_id="health_preprocessing",
    schedule=None,  # run on demand when new raw data arrives
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=TAGS,
    doc_md=__doc__,
)
def health_preprocessing():
    # Project imports stay inside tasks so DAG parsing stays fast and does not need the ML libraries

    @task
    def download_raw_data() -> str:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.preprocessing import download_raw_data

        return str(download_raw_data(load_config()))

    @task
    def clean_dataset(raw_path: str) -> str:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.preprocessing import run_preprocessing

        return str(run_preprocessing(load_config()))

    @task(outlets=[CLEANED_DATASET])
    def validate_cleaned_data(cleaned_path: str) -> dict:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.preprocessing import validate_cleaned_data

        return validate_cleaned_data(load_config())

    validate_cleaned_data(clean_dataset(download_raw_data()))


health_preprocessing()
