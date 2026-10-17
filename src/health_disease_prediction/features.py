"""Encoding and scaling (same steps as 03_training.ipynb).

Every encoder is fitted on training rows only; the fitted objects travel together in one
`preprocessors` dict so training and inference always apply an identical transformation.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def fit_preprocessors(X_train: pd.DataFrame, features: dict) -> dict:
    # Nominal categories have no order, so each value gets its own 0/1 column; unseen values become all zeros
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    one_hot_encoder.fit(X_train[features["one_hot"]])

    # Blood_Pressure has hundreds of distinct readings, too many for one-hot, so each reading is encoded by how often it occurs
    frequency_maps = {col: X_train[col].value_counts(normalize=True).to_dict() for col in features["frequency"]}

    # Follow_Up has two ordered values (14, 30 days); OrdinalEncoder is the feature-side equivalent of LabelEncoder
    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ordinal_encoder.fit(X_train[features["ordinal"]])

    # Put numeric features on one scale so distance and weight based models (SVM, LR) are not dominated by large values
    scaler = StandardScaler()
    scaler.fit(X_train[features["scale"]])

    preprocessors = {
        "one_hot_encoder": one_hot_encoder,
        "frequency_maps": frequency_maps,
        "ordinal_encoder": ordinal_encoder,
        "standard_scaler": scaler,
    }
    # Column order the models are trained on; inference input must match it exactly
    preprocessors["feature_columns"] = list(transform_features(X_train, preprocessors, features).columns)
    return preprocessors


def transform_features(X: pd.DataFrame, preprocessors: dict, features: dict) -> pd.DataFrame:
    scaler = preprocessors["standard_scaler"]
    ordinal_encoder = preprocessors["ordinal_encoder"]
    one_hot_encoder = preprocessors["one_hot_encoder"]
    frequency_maps = preprocessors["frequency_maps"]

    scaled = pd.DataFrame(scaler.transform(X[features["scale"]]), columns=features["scale"], index=X.index)
    ordinal = pd.DataFrame(ordinal_encoder.transform(X[features["ordinal"]]), columns=features["ordinal"], index=X.index)
    # Readings never seen in training get frequency 0
    frequency = pd.DataFrame(
        {f"{col}_freq": X[col].map(frequency_maps[col]).fillna(0.0) for col in features["frequency"]},
        index=X.index,
    )
    one_hot = pd.DataFrame(
        one_hot_encoder.transform(X[features["one_hot"]]),
        columns=one_hot_encoder.get_feature_names_out(),
        index=X.index,
    )
    encoded = pd.concat([scaled, ordinal, frequency, one_hot], axis=1)

    expected = preprocessors.get("feature_columns")
    if expected is not None and list(encoded.columns) != expected:
        raise ValueError("Encoded columns differ from the columns the model was trained on")
    return encoded
