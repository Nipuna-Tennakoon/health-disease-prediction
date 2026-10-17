import pandas as pd
import pytest

from health_disease_prediction.features import fit_preprocessors, transform_features

FEATURES = {
    "one_hot": ["City"],
    "frequency": ["Blood_Pressure"],
    "ordinal": ["Follow_Up"],
    "scale": ["Age"],
}


@pytest.fixture
def train():
    return pd.DataFrame({
        "City": ["Boston", "Chicago", "Boston"],
        "Blood_Pressure": ["120/80", "120/80", "130/85"],
        "Follow_Up": [14.0, 30.0, 14.0],
        "Age": [20.0, 40.0, 60.0],
    })


def test_transform_uses_training_statistics(train):
    preprocessors = fit_preprocessors(train, FEATURES)
    encoded = transform_features(train, preprocessors, FEATURES)

    assert list(encoded.columns) == preprocessors["feature_columns"]
    assert encoded["Age"].mean() == pytest.approx(0.0)
    assert encoded.loc[0, "Blood_Pressure_freq"] == pytest.approx(2 / 3)


def test_unseen_values_do_not_break_inference(train):
    preprocessors = fit_preprocessors(train, FEATURES)
    new = pd.DataFrame({"City": ["Denver"], "Blood_Pressure": ["999/99"], "Follow_Up": [7.0], "Age": [30.0]})
    encoded = transform_features(new, preprocessors, FEATURES)

    assert encoded.filter(like="City_").sum(axis=1).iloc[0] == 0
    assert encoded.loc[0, "Blood_Pressure_freq"] == 0.0
    assert encoded.loc[0, "Follow_Up"] == -1


def test_column_mismatch_is_rejected(train):
    preprocessors = fit_preprocessors(train, FEATURES)
    preprocessors["feature_columns"] = list(reversed(preprocessors["feature_columns"]))
    with pytest.raises(ValueError, match="differ"):
        transform_features(train, preprocessors, FEATURES)
