import numpy as np
import pandas as pd
import pytest

from health_disease_prediction.evaluation import (
    corrected_resampled_ttest,
    holm_adjust,
    pairwise_comparison,
    select_model,
)


def test_correction_makes_the_test_more_conservative():
    a = [0.5337, 0.4966, 0.4999, 0.4903, 0.5207]
    b = [0.5035, 0.4783, 0.4945, 0.4825, 0.4925]
    mean_diff, t_corrected, p_corrected = corrected_resampled_ttest(a, b, test_train_ratio=0.25)
    _, t_naive, p_naive = corrected_resampled_ttest(a, b, test_train_ratio=0.0)

    assert mean_diff == pytest.approx(0.0180, abs=1e-4)
    assert abs(t_corrected) < abs(t_naive)
    assert p_corrected > p_naive


def test_identical_scores_give_no_evidence():
    assert corrected_resampled_ttest([0.5] * 5, [0.5] * 5, 0.25) == (0.0, 0.0, 1.0)


def test_holm_adjustment():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def _significance(fold_scores):
    return pairwise_comparison(fold_scores, test_train_ratio=0.25, alpha=0.05)


def test_no_model_beats_baseline_selects_simplest():
    rng = np.random.default_rng(0)
    fold_scores = {
        "xgboost": list(0.5 + rng.normal(0, 0.01, 25)),
        "logistic_regression": list(0.5 + rng.normal(0, 0.01, 25)),
        "baseline": [0.5] * 25,
    }
    means = {k: float(np.mean(v)) for k, v in fold_scores.items()}
    decision = select_model(means, _significance(fold_scores), "baseline", ["logistic_regression", "xgboost"])

    assert decision["significant"] is False
    assert decision["name"] == "logistic_regression"


def test_clear_winner_is_selected_and_significant():
    rng = np.random.default_rng(0)
    fold_scores = {
        "xgboost": list(0.75 + rng.normal(0, 0.01, 25)),
        "logistic_regression": list(0.5 + rng.normal(0, 0.01, 25)),
        "baseline": [0.5] * 25,
    }
    means = {k: float(np.mean(v)) for k, v in fold_scores.items()}
    decision = select_model(means, _significance(fold_scores), "baseline", ["logistic_regression", "xgboost"])

    assert decision["significant"] is True
    assert decision["beats_baseline"] == ["xgboost"]
    assert decision["name"] == "xgboost"


def test_pairwise_table_covers_every_pair():
    table = _significance({"a": [0.5, 0.6, 0.55], "b": [0.5, 0.5, 0.5], "c": [0.4, 0.45, 0.5]})
    assert len(table) == 3
    assert set(table.columns) >= {"mean_diff", "t_stat", "p_value", "p_holm", "significant"}
    assert isinstance(table, pd.DataFrame)
