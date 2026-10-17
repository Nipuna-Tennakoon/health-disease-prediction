"""Load config.yml and resolve project paths.

The project root defaults to the repository that contains this package. Set HDP_PROJECT_ROOT
(and optionally HDP_CONFIG_PATH) when the code runs elsewhere, e.g. inside an Airflow worker.
"""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.getenv("HDP_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(os.getenv("HDP_CONFIG_PATH", PROJECT_ROOT / "config.yml"))


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Resolve a config path (relative to the project root) to an absolute path."""
    return PROJECT_ROOT / relative


def artifact_paths(cfg: dict) -> dict[str, Path]:
    """Versioned artifact files shared by preprocessing, training and inference."""
    version = cfg["experiment"]["version"]
    encoders_dir = project_path(cfg["artifacts"]["encoders_dir"])
    return {
        "fill_values": encoders_dir / f"fill_values_{version}.json",
        "preprocessors": encoders_dir / f"preprocessors_{version}.joblib",
        "models": project_path(cfg["artifacts"]["models_dir"]) / f"models_{version}.joblib",
    }
