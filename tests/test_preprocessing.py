import numpy as np
import pandas as pd
import pytest

from health_disease_prediction import preprocessing as pp


@pytest.mark.parametrize(
    ("cleaner", "raw", "expected"),
    [
        (pp.clean_age, "twenty", 20),
        (pp.clean_age, "250", 25.0),
        (pp.clean_age, "-5", np.nan),
        (pp.clean_gender, "FEMALE", "F"),
        (pp.clean_gender, "Unknown", np.nan),
        (pp.clean_city, "Nwe Yrok", "New York"),
        (pp.clean_city, "Los Angeles", "Los Angeles"),
        (pp.clean_bmi, "22kg/m2", 22.0),
        (pp.clean_blood_pressure, "120 over 80", "120/80"),
        (pp.clean_blood_pressure, "120 - 80", "120/80"),
        (pp.clean_heart_rate, "500", np.nan),
        (pp.clean_heart_rate, "eighty", np.nan),
        (pp.clean_cholesterol, "high", "250"),
        (pp.clean_smoker, "EX-smoker", "Ex"),
        (pp.clean_medications, "Aspirin; Metformin", "aspirin,metformin"),
        (pp.clean_follow_up, "two weeks", np.nan),
        (pp.clean_target, "unknown", np.nan),
        (pp.clean_target, "1", 1),
    ],
)
def test_cleaners_handle_raw_formats(cleaner, raw, expected):
    result = cleaner(raw)
    if isinstance(expected, float) and np.isnan(expected):
        assert pd.isna(result)
    else:
        assert result == expected


def test_patient_id_gaps_are_rebuilt_from_row_position():
    ids = pd.Series(["ID-00001", np.nan, "ID-00003"], index=[0, 1, 2])
    assert pp.clean_patient_id(ids).tolist() == ["ID-1", "ID-2", "ID-3"]


def test_fill_values_are_learned_once_and_reused():
    train = pd.DataFrame({col: ["x"] for col in pp.FEATURE_CLEANERS})
    train[["Age", "BMI", "Heart_Rate", "Cholesterol_Level", "Follow_Up"]] = [[30.0, 20.0, 70.0, "190", 14]]
    fill_values = pp.compute_fill_values(train)

    new = pd.DataFrame({col: [np.nan] for col in pp.FEATURE_CLEANERS})
    filled = pp.fill_missing(new, fill_values)

    assert not filled.isna().any().any()
    assert filled.loc[0, "Age"] == 30.0
    assert filled.loc[0, "Medications"] == "none"
    assert filled["Cholesterol_Level"].dtype == float
