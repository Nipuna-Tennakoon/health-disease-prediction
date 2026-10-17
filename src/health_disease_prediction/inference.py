"""Predict on new raw records with the saved artifacts (same logic as 04_inference.ipynb)."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from health_disease_prediction.config import artifact_paths, project_path
from health_disease_prediction.features import transform_features
from health_disease_prediction.preprocessing import (
    FEATURE_CLEANERS,
    clean_columns,
    fill_missing,
    load_fill_values,
)

logger = logging.getLogger(__name__)


def generate_synthetic_data(cfg: dict) -> Path:
    """Sample every raw column independently, so new rows carry the same messy formats the cleaning must handle.

    The target is left out because it is what we predict. IDs get an S prefix so synthetic
    patients can never be confused with real ones.
    """
    target = cfg["data"]["target"]
    n_samples = cfg["inference"]["n_samples"]
    raw = pd.read_csv(project_path(cfg["data"]["raw_path"]), dtype=str, keep_default_na=False)
    rng = np.random.default_rng(cfg["data"]["random_state"])

    sample_columns = [c for c in raw.columns if c not in ["Patient_ID", target]]
    synthetic = pd.DataFrame({col: rng.choice(raw[col].to_numpy(), size=n_samples) for col in sample_columns})
    synthetic.insert(0, "Patient_ID", [f"ID-S{i:03d}" for i in range(1, n_samples + 1)])

    path = project_path(cfg["inference"]["synthetic_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(path, index=False)
    logger.info("Saved %d synthetic records to %s", n_samples, path)
    return path


def validate_input(df: pd.DataFrame) -> None:
    """Fail fast if a column the cleaning rules expect is missing."""
    missing = {"Patient_ID", *FEATURE_CLEANERS} - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")


def preprocess_for_inference(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    """Clean and fill model features with the values learned during preprocessing; keep Patient_ID to label results."""
    clean = fill_missing(clean_columns(df[["Patient_ID", *FEATURE_CLEANERS]]), fill_values)
    if clean.isna().any().any():
        raise ValueError("Missing values remain after cleaning")
    return clean


def load_best_model(cfg: dict):
    """Load the model named in config.yml and check its version against the saved bundle."""
    best = cfg["experiment"]["best_model"]
    name, version = best["name"], str(best["version"])
    models_path = artifact_paths(cfg)["models"]
    models = joblib.load(models_path)

    if name not in models:
        raise KeyError(f"{name} not found in {models_path.name}")
    if models[name]["version"] != version:
        raise ValueError(f"config.yml expects {name} v{version}, bundle has v{models[name]['version']}")
    logger.info("Loaded %s v%s from %s", name, version, models_path.name)
    return models[name]["model"]


def predict(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Raw records -> predicted class and probability of disease."""
    validate_input(df)
    paths = artifact_paths(cfg)
    clean = preprocess_for_inference(df, load_fill_values(paths["fill_values"]))

    preprocessors = joblib.load(paths["preprocessors"])
    X_enc = transform_features(clean, preprocessors, cfg["features"])
    model = load_best_model(cfg)

    predictions = pd.DataFrame({
        "Patient_ID": clean["Patient_ID"],
        "predicted_class": model.predict(X_enc),
        "probability_disease": model.predict_proba(X_enc)[:, 1].round(3),
    })
    predictions["predicted_label"] = predictions["predicted_class"].map({0: "no disease", 1: "disease"})
    return predictions


def run_inference(cfg: dict, input_path: Path | None = None) -> Path:
    """Predict on a raw CSV (the synthetic test data by default) and save the predictions."""
    input_path = input_path or project_path(cfg["inference"]["synthetic_path"])
    # Read the file the same way preprocessing reads the raw data, so placeholders are handled identically
    predictions = predict(pd.read_csv(input_path), cfg)

    output_path = project_path(cfg["inference"]["predictions_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Saved %d predictions to %s", len(predictions), output_path)
    return output_path
