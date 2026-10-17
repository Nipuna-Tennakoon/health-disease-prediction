"""Split, encode, train, compare and select models (same logic as 03_training.ipynb).

Each public function is one pipeline step that reads its inputs from disk and writes its outputs
to disk, so steps can run in separate processes (Airflow tasks) or in sequence via run_training.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import wandb
import xgboost
from dotenv import load_dotenv
from ruamel.yaml import YAML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from health_disease_prediction.config import CONFIG_PATH, PROJECT_ROOT, artifact_paths, project_path
from health_disease_prediction.features import fit_preprocessors, transform_features

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "xgboost": XGBClassifier,
}


def _split_paths(cfg: dict) -> dict:
    splits_dir = project_path(cfg["data"]["splits_dir"])
    return {"train": splits_dir / "train.csv", "test": splits_dir / "test.csv"}


def _staging_paths(cfg: dict, model_name: str) -> dict:
    staging_dir = project_path(cfg["artifacts"]["staging_dir"])
    return {"model": staging_dir / f"{model_name}.joblib", "metrics": staging_dir / f"{model_name}_metrics.json"}


def load_split(cfg: dict, split: str) -> tuple[pd.DataFrame, pd.Series]:
    target = cfg["data"]["target"]
    df = pd.read_csv(_split_paths(cfg)[split])
    return df.drop(columns=target), df[target]


def split_dataset(cfg: dict) -> dict:
    """Stratified train/test split of the cleaned dataset; the fixed seed makes it reproducible."""
    target = cfg["data"]["target"]
    df = pd.read_csv(project_path(cfg["data"]["cleaned_path"]))
    # Identifiers and the raw visit date carry no generalisable signal for a new patient
    df = df.drop(columns=cfg["data"]["drop_columns"])

    train, test = train_test_split(
        df,
        test_size=cfg["data"]["test_size"],
        stratify=df[target],
        random_state=cfg["data"]["random_state"],
    )
    paths = _split_paths(cfg)
    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    logger.info("Split %d rows into train %s and test %s", len(df), train.shape, test.shape)
    return {"train_rows": len(train), "test_rows": len(test)}


def fit_and_save_preprocessors(cfg: dict) -> str:
    """Fit encoders and scaler on the training split only (no test leakage) and save them as one file."""
    X_train, _ = load_split(cfg, "train")
    preprocessors = fit_preprocessors(X_train, cfg["features"])
    path = artifact_paths(cfg)["preprocessors"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessors, path)
    logger.info("Saved preprocessors (%d encoded features) to %s", len(preprocessors["feature_columns"]), path)
    return str(path)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    # ROC-AUC only needs a ranking score; SVC's predict_proba comes from an internal CV calibration
    # that can disagree with its own predictions, so prefer decision_function when the model has one
    scores = model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
    }


def wandb_settings(cfg: dict) -> dict:
    """Shared W&B settings; the API key and team come from .env, never from code or config."""
    load_dotenv(PROJECT_ROOT / cfg["wandb"]["env_file"])
    if cfg["wandb"]["mode"] == "online":
        api_key = os.getenv(cfg["wandb"]["api_key_env"])
        if not api_key:
            raise ValueError(f"Set {cfg['wandb']['api_key_env']} in {cfg['wandb']['env_file']} or change wandb.mode")
        wandb.login(key=api_key)
    return {
        "project": cfg["wandb"]["project"],
        "entity": os.getenv(cfg["wandb"]["entity_env"]),
        "mode": cfg["wandb"]["mode"],
        "dir": PROJECT_ROOT / cfg["wandb"]["dir"],
        "group": cfg["experiment"]["version"],
    }


def train_model(cfg: dict, model_name: str) -> dict:
    """Train one model, log train/test metrics to W&B and stage the fitted model for bundling."""
    spec = cfg["models"][model_name]
    features = cfg["features"]
    preprocessors = joblib.load(artifact_paths(cfg)["preprocessors"])

    X_train, y_train = load_split(cfg, "train")
    X_test, y_test = load_split(cfg, "test")
    X_train_enc = transform_features(X_train, preprocessors, features)
    X_test_enc = transform_features(X_test, preprocessors, features)

    model = MODEL_CLASSES[model_name](**spec["params"])
    model.fit(X_train_enc, y_train)
    train_metrics = evaluate(model, X_train_enc, y_train)
    test_metrics = evaluate(model, X_test_enc, y_test)

    # One W&B run per model, grouped by experiment version, so train vs test metrics can be compared side by side
    run = wandb.init(
        **wandb_settings(cfg),
        name=f"{model_name}-v{spec['version']}",
        job_type="train",
        config={
            "model": model_name,
            "model_version": spec["version"],
            "params": spec["params"],
            "features": features,
            "n_encoded_features": X_train_enc.shape[1],
            "train_rows": len(X_train_enc),
            "test_rows": len(X_test_enc),
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
        },
    )
    run.log({
        **{f"train/{k}": v for k, v in train_metrics.items()},
        **{f"test/{k}": v for k, v in test_metrics.items()},
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=model.predict(X_test_enc).tolist(),
            class_names=["no disease", "disease"],
        ),
    })
    run.finish()

    result = {
        "model": model_name,
        "version": spec["version"],
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    paths = _staging_paths(cfg, model_name)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": spec["version"], "model": model}, paths["model"])
    paths["metrics"].write_text(json.dumps(result, indent=2))
    logger.info("%s: test accuracy %.3f, test ROC-AUC %.3f", model_name, test_metrics["accuracy"], test_metrics["roc_auc"])
    return result


def package_models(cfg: dict) -> pd.DataFrame:
    """Bundle the staged models into one versioned file, save the comparison table and log W&B artifacts."""
    trained_models, results = {}, []
    for model_name in cfg["models"]:
        paths = _staging_paths(cfg, model_name)
        trained_models[model_name] = joblib.load(paths["model"])
        results.append(json.loads(paths["metrics"].read_text()))

    paths = artifact_paths(cfg)
    paths["models"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models, paths["models"])

    comparison = pd.DataFrame(results).set_index("model").round(4)
    metrics_path = project_path(cfg["artifacts"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_path)

    # Log both files as W&B artifacts, tagged with the experiment version, so each run is traceable to its exact files
    version = cfg["experiment"]["version"]
    run = wandb.init(**wandb_settings(cfg), name=f"artifacts-{version}", job_type="upload")
    for path, artifact_type in [(paths["preprocessors"], "preprocessor"), (paths["models"], "model")]:
        artifact = wandb.Artifact(path.stem.removesuffix(f"_{version}"), type=artifact_type, metadata={"experiment_version": version})
        artifact.add_file(str(path))
        run.log_artifact(artifact, aliases=[version])
    run.finish()

    logger.info("Saved %d models to %s and comparison to %s", len(trained_models), paths["models"], metrics_path)
    return comparison


def select_best_model(cfg: dict, config_path: Path = CONFIG_PATH) -> dict:
    """Pick the model with the highest selection metric and record it in config.yml for inference."""
    comparison = pd.read_csv(project_path(cfg["artifacts"]["metrics_path"]), index_col="model")
    metric = cfg["experiment"]["selection_metric"]
    best_name = comparison[metric].idxmax()
    best = {"name": best_name, "version": str(comparison.loc[best_name, "version"])}

    # Round-trip YAML keeps the comments and key order of config.yml when writing it back
    yaml_rt = YAML()
    config_doc = yaml_rt.load(config_path)
    config_doc["experiment"]["best_model"]["name"] = best["name"]
    config_doc["experiment"]["best_model"]["version"] = best["version"]
    yaml_rt.dump(config_doc, config_path)

    logger.info("Best model by %s: %s v%s (%.4f)", metric, best["name"], best["version"], comparison.loc[best_name, metric])
    return best


def run_training(cfg: dict, config_path: Path = CONFIG_PATH) -> dict:
    """All training steps in sequence (what the Airflow training DAG runs as separate tasks)."""
    split_dataset(cfg)
    fit_and_save_preprocessors(cfg)
    for model_name in cfg["models"]:
        train_model(cfg, model_name)
    package_models(cfg)
    return select_best_model(cfg, config_path)
