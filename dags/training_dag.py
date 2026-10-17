"""
### Health training
Triggered by the **cleaned_dataset** asset.

1. Stratified train/test split of the cleaned dataset
2. Fit encoders and scaler on the training split and save them as one versioned file
3. For every model in `config.yml`, in parallel (one mapped task per model): 5-fold cross-validation on the
   training split, then a final fit on the full training split; fold scores, CV mean/std and test metrics go to W&B
4. Bundle the models into one versioned file and save the comparison table
5. Write the best model (highest CV mean) to `config.yml`, publishing the **trained_models** asset that triggers inference
"""

from datetime import datetime

from airflow.sdk import dag, task

from health_disease_prediction.orchestration import CLEANED_DATASET, DEFAULT_ARGS, TAGS, TRAINED_MODELS


@dag(
    dag_id="health_training",
    schedule=[CLEANED_DATASET],
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=TAGS,
    doc_md=__doc__,
)
def health_training():
    # Project imports stay inside tasks so DAG parsing stays fast and does not need the ML libraries

    @task
    def split_dataset() -> dict:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.training import split_dataset

        return split_dataset(load_config())

    @task
    def fit_preprocessors() -> str:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.training import fit_and_save_preprocessors

        return fit_and_save_preprocessors(load_config())

    @task
    def list_models() -> list[str]:
        from health_disease_prediction.config import load_config

        return list(load_config()["models"])

    @task
    def train_model(model_name: str) -> str:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.training import train_model

        train_model(load_config(), model_name)
        return model_name

    @task
    def package_models() -> None:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.training import package_models

        package_models(load_config())

    @task(outlets=[TRAINED_MODELS])
    def select_best_model() -> dict:
        from health_disease_prediction.config import load_config
        from health_disease_prediction.training import select_best_model

        return select_best_model(load_config())

    trained = train_model.expand(model_name=list_models())
    split_dataset() >> fit_preprocessors() >> trained >> package_models() >> select_best_model()


health_training()
