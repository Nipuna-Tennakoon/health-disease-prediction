"""Split, encode, cross-validate, train, compare and select models (same logic as 03_training.ipynb).

Each public function is one pipeline step that reads its inputs from disk and writes its outputs
to disk, so steps can run in separate processes (Airflow tasks) or in sequence via run_training.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from dotenv import load_dotenv
from ruamel.yaml import YAML
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

import wandb
from health_disease_prediction.config import (
    CONFIG_PATH,
    PROJECT_ROOT,
    artifact_paths,
    project_path,
)
from health_disease_prediction.evaluation import pairwise_comparison, select_model
from health_disease_prediction.features import fit_preprocessors, transform_features

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "xgboost": XGBClassifier,
    # No-skill reference: predicts class frequencies only, so it scores ROC-AUC 0.5 on every fold
    "baseline": DummyClassifier,
}
BASELINE = "baseline"


def _split_paths(cfg: dict) -> dict:
    splits_dir = project_path(cfg["data"]["splits_dir"])
    return {"train": splits_dir / "train.csv", "test": splits_dir / "test.csv"}


def _staging_paths(cfg: dict, model_name: str) -> dict:
    staging_dir = project_path(cfg["artifacts"]["staging_dir"])
    return {
        "model": staging_dir / f"{model_name}.joblib",
        "metrics": staging_dir / f"{model_name}_metrics.json",
        "folds": staging_dir / f"{model_name}_folds.json",
    }


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


def cross_validate(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    features: dict,
    n_splits: int,
    random_state: int,
    n_repeats: int = 1,
) -> list[dict]:
    """Repeated stratified k-fold on the training split only; the test set is never touched here.

    Each repeat reshuffles the rows into new folds. The fixed seed gives every model exactly the same
    folds, so their scores can be compared fold by fold. Encoders and scaler are refitted inside every
    fold on that fold's training part, so the held-out fold is as unseen as real new data would be.
    Scores are returned in order: repeat 1 folds 1..k, repeat 2 folds 1..k, ...
    """
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_metrics = []
    for fit_idx, val_idx in folds.split(X, y):
        X_fit, X_val = X.iloc[fit_idx], X.iloc[val_idx]
        y_fit, y_val = y.iloc[fit_idx], y.iloc[val_idx]

        fold_preprocessors = fit_preprocessors(X_fit, features)
        model = MODEL_CLASSES[model_name](**params)
        model.fit(transform_features(X_fit, fold_preprocessors, features), y_fit)
        fold_metrics.append(evaluate(model, transform_features(X_val, fold_preprocessors, features), y_val))
    return fold_metrics


def summarize_folds(fold_metrics: list[dict]) -> dict:
    """Mean and standard deviation of each metric across folds."""
    summary = {}
    for metric in fold_metrics[0]:
        values = [fold[metric] for fold in fold_metrics]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
    return summary


def fold_positions(n_scores: int, n_splits: int) -> list[tuple[int, int]]:
    """(repeat, fold) number of each score, in the order cross_validate returns them."""
    return [(i // n_splits + 1, i % n_splits + 1) for i in range(n_scores)]


def log_folds(run, fold_metrics: list[dict], cv_summary: dict, n_splits: int) -> None:
    """Per-fold scores as a chart and a table, plus mean/std in the run summary for comparing models."""
    run.define_metric("split")
    run.define_metric("cv/*", step_metric="split")
    fold_table = wandb.Table(columns=["repeat", "fold", *fold_metrics[0]])
    positions = fold_positions(len(fold_metrics), n_splits)
    for split, (metrics, (repeat, fold)) in enumerate(zip(fold_metrics, positions), start=1):
        run.log({"split": split, **{f"cv/{k}": v for k, v in metrics.items()}})
        fold_table.add_data(repeat, fold, *metrics.values())
    run.log({"cv/folds": fold_table})
    for key, value in cv_summary.items():
        run.summary[f"cv/{key}"] = value


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
    """Cross-validate one model, refit it on the full training split, log everything to W&B and stage it."""
    spec = cfg["models"][model_name]
    features = cfg["features"]
    preprocessors = joblib.load(artifact_paths(cfg)["preprocessors"])

    X_train, y_train = load_split(cfg, "train")
    X_test, y_test = load_split(cfg, "test")
    X_train_enc = transform_features(X_train, preprocessors, features)
    X_test_enc = transform_features(X_test, preprocessors, features)

    # Repeated cross-validation estimates how good and how stable the model is; it drives model selection
    fold_metrics = cross_validate(
        model_name, spec["params"], X_train, y_train, features,
        n_splits=cfg["cv"]["n_splits"], random_state=cfg["data"]["random_state"], n_repeats=cfg["cv"]["n_repeats"],
    )
    cv_summary = summarize_folds(fold_metrics)

    # Final model is refitted on the whole training split; the test set is scored once, for reporting
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
            "cv_folds": cfg["cv"]["n_splits"],
            "cv_repeats": cfg["cv"]["n_repeats"],
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
    log_folds(run, fold_metrics, cv_summary, cfg["cv"]["n_splits"])
    run.finish()

    result = {
        "model": model_name,
        "version": spec["version"],
        **{f"cv_{k}": v for k, v in cv_summary.items()},
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    paths = _staging_paths(cfg, model_name)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": spec["version"], "model": model}, paths["model"])
    paths["metrics"].write_text(json.dumps(result, indent=2))
    # Per-fold scores are kept for the statistical comparison, which needs every model's scores on the same folds
    paths["folds"].write_text(json.dumps(fold_metrics, indent=2))
    logger.info(
        "%s: CV ROC-AUC %.3f ± %.3f, test ROC-AUC %.3f",
        model_name, cv_summary["roc_auc_mean"], cv_summary["roc_auc_std"], test_metrics["roc_auc"],
    )
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


def compare_models(cfg: dict) -> pd.DataFrame:
    """Test every pair of models (and a no-skill baseline) for a real difference in their CV scores.

    The baseline is cross-validated on the same folds as the models. Pairs are compared with the
    corrected resampled t-test and Holm-adjusted p-values; the fold scores and the significance
    table are saved to reports/ and logged to W&B.
    """
    n_splits = cfg["cv"]["n_splits"]
    metric = cfg["selection"]["metric"]
    fold_metrics = {name: json.loads(_staging_paths(cfg, name)["folds"].read_text()) for name in cfg["models"]}

    X_train, y_train = load_split(cfg, "train")
    fold_metrics[BASELINE] = cross_validate(
        BASELINE, {"strategy": cfg["selection"]["baseline_strategy"]}, X_train, y_train, cfg["features"],
        n_splits=n_splits, random_state=cfg["data"]["random_state"], n_repeats=cfg["cv"]["n_repeats"],
    )

    fold_scores = pd.DataFrame([
        {"model": name, "repeat": repeat, "fold": fold, **scores}
        for name, folds in fold_metrics.items()
        for scores, (repeat, fold) in zip(folds, fold_positions(len(folds), n_splits))
    ])
    # Held-out fold vs training part size: 1/(k-1), the overlap correction in the t-test
    significance = pairwise_comparison(
        {name: [fold[metric] for fold in folds] for name, folds in fold_metrics.items()},
        test_train_ratio=1 / (n_splits - 1),
        alpha=cfg["selection"]["alpha"],
    )

    fold_scores_path = project_path(cfg["artifacts"]["fold_scores_path"])
    significance_path = project_path(cfg["artifacts"]["significance_path"])
    fold_scores_path.parent.mkdir(parents=True, exist_ok=True)
    fold_scores.round(4).to_csv(fold_scores_path, index=False)
    significance.round(4).to_csv(significance_path, index=False)

    version = cfg["experiment"]["version"]
    run = wandb.init(**wandb_settings(cfg), name=f"comparison-{version}", job_type="evaluate",
                     config={"metric": metric, "alpha": cfg["selection"]["alpha"],
                             "cv_folds": n_splits, "cv_repeats": cfg["cv"]["n_repeats"]})
    run.log({"comparison/significance": wandb.Table(dataframe=significance.round(4)),
             "comparison/fold_scores": wandb.Table(dataframe=fold_scores.round(4))})
    run.finish()

    logger.info("Compared %d models on %d folds: %d of %d pairs significant after Holm correction",
                len(fold_metrics), len(fold_scores) // len(fold_metrics),
                int(significance["significant"].sum()), len(significance))
    return significance


def select_best_model(cfg: dict, config_path: Path = CONFIG_PATH) -> dict:
    """Apply the statistical selection rule and record the chosen model in config.yml for inference."""
    metric = cfg["selection"]["metric"]
    fold_scores = pd.read_csv(project_path(cfg["artifacts"]["fold_scores_path"]))
    significance = pd.read_csv(project_path(cfg["artifacts"]["significance_path"]))
    comparison = pd.read_csv(project_path(cfg["artifacts"]["metrics_path"]), index_col="model")

    means = fold_scores.groupby("model")[metric].mean().to_dict()
    decision = select_model(means, significance, BASELINE, cfg["selection"]["simplicity_order"])
    best = {
        "name": decision["name"],
        "version": str(comparison.loc[decision["name"], "version"]),
        "significant": decision["significant"],
    }

    # Round-trip YAML keeps the comments and key order of config.yml when writing it back
    yaml_rt = YAML()
    config_doc = yaml_rt.load(config_path)
    for key, value in best.items():
        config_doc["experiment"]["best_model"][key] = value
    yaml_rt.dump(config_doc, config_path)

    if decision["significant"]:
        logger.info("Selected %s v%s: beats the baseline significantly; tied with top model: %s",
                    best["name"], best["version"], decision["tied_with_top"])
    else:
        logger.warning("No model beats the no-skill baseline significantly; selected the simplest equivalent model "
                       "%s v%s (top by mean: %s)", best["name"], best["version"], decision["top_by_mean"])
    return {**best, "top_by_mean": decision["top_by_mean"], "tied_with_top": decision["tied_with_top"]}


def run_training(cfg: dict, config_path: Path = CONFIG_PATH) -> dict:
    """All training steps in sequence (what the Airflow training DAG runs as separate tasks)."""
    split_dataset(cfg)
    fit_and_save_preprocessors(cfg)
    for model_name in cfg["models"]:
        train_model(cfg, model_name)
    package_models(cfg)
    compare_models(cfg)
    return select_best_model(cfg, config_path)
