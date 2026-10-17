"""Command-line entry point: `hdp <stage>` runs one pipeline stage outside Airflow."""

import argparse
import logging
from pathlib import Path

from health_disease_prediction import inference, preprocessing, training
from health_disease_prediction.config import CONFIG_PATH, load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="hdp", description="Health disease prediction pipeline")
    parser.add_argument("stage", choices=["preprocess", "train", "predict", "all"])
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--input", type=Path, help="raw CSV to predict on (default: generated synthetic data)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.stage in ("preprocess", "all"):
        preprocessing.download_raw_data(cfg)
        preprocessing.run_preprocessing(cfg)
        preprocessing.validate_cleaned_data(cfg)
    if args.stage in ("train", "all"):
        training.run_training(cfg, args.config)
        cfg = load_config(args.config)  # pick up the best model training just wrote
    if args.stage in ("predict", "all"):
        if args.input is None:
            inference.generate_synthetic_data(cfg)
        inference.run_inference(cfg, args.input)


if __name__ == "__main__":
    main()
