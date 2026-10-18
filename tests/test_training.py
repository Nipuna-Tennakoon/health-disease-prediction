import numpy as np
import pandas as pd
import pytest

from health_disease_prediction.training import cross_validate, summarize_folds

FEATURES = {
    "one_hot": ["City"],
    "frequency": ["Blood_Pressure"],
    "ordinal": ["Follow_Up"],
    "scale": ["Age"],
}


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({
        "City": rng.choice(["Boston", "Chicago"], n),
        "Blood_Pressure": rng.choice(["120/80", "130/85"], n),
        "Follow_Up": rng.choice([14.0, 30.0], n),
        "Age": rng.normal(40, 10, n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_cross_validate_returns_one_score_set_per_fold(data):
    X, y = data
    folds = cross_validate("logistic_regression", {"max_iter": 200}, X, y, FEATURES, n_splits=5, random_state=42)

    assert len(folds) == 5
    assert set(folds[0]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_repeated_cross_validation_returns_folds_times_repeats(data):
    X, y = data
    folds = cross_validate(
        "logistic_regression", {"max_iter": 200}, X, y, FEATURES, n_splits=5, random_state=42, n_repeats=3
    )
    assert len(folds) == 15


def test_baseline_has_no_skill(data):
    X, y = data
    folds = cross_validate("baseline", {"strategy": "prior"}, X, y, FEATURES, n_splits=5, random_state=42, n_repeats=2)
    assert all(fold["roc_auc"] == 0.5 for fold in folds)


def test_cross_validate_is_reproducible(data):
    X, y = data
    run = lambda: cross_validate("random_forest", {"n_estimators": 10, "random_state": 0}, X, y, FEATURES, 5, 42)
    assert run() == run()


def test_summarize_folds_gives_mean_and_std():
    summary = summarize_folds([{"roc_auc": 0.4}, {"roc_auc": 0.6}])
    assert summary == {"roc_auc_mean": pytest.approx(0.5), "roc_auc_std": pytest.approx(0.1)}
