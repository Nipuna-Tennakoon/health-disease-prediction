"""Streamlit demo: score raw patient records with the model selected by training.

All cleaning, encoding and model loading come from the health_disease_prediction package,
so the app shows exactly what the pipeline does. Run from the project root:

    uv run --group app streamlit run app/streamlit_app.py
"""

import io

import pandas as pd
import streamlit as st

from health_disease_prediction.config import load_config, project_path
from health_disease_prediction.inference import (
    load_artifacts,
    predict,
    preprocess_for_inference,
)
from health_disease_prediction.preprocessing import FEATURE_CLEANERS, clean_columns

FEATURES = list(FEATURE_CLEANERS)

st.set_page_config(page_title="Health Disease Prediction", page_icon="🩺", layout="wide")


@st.cache_resource
def get_model() -> tuple[dict, dict]:
    """Config and artifacts are loaded once per app session; restart the app after retraining."""
    cfg = load_config()
    return cfg, load_artifacts(cfg)


@st.cache_data
def load_examples() -> pd.DataFrame:
    """Synthetic records as raw strings, so the form shows values exactly as they appear in the data."""
    cfg, _ = get_model()
    return pd.read_csv(project_path(cfg["inference"]["synthetic_path"]), dtype=str, keep_default_na=False)


def as_pipeline_input(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through CSV so form input is parsed exactly like a raw file (blanks, N/A, None -> missing)."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def fill_random_example() -> None:
    row = load_examples().sample(1).iloc[0]
    for col in FEATURES:
        st.session_state[f"field_{col}"] = row[col]


cfg, artifacts = get_model()
best = cfg["experiment"]["best_model"]

st.title("🩺 Health Disease Prediction")
st.caption(f"Model: **{best['name']} v{best['version']}** (experiment {cfg['experiment']['version']})")
if not best.get("significant", False):
    st.warning(
        "No model outperformed a no-skill baseline in cross-validation, so these predictions are no better "
        "than chance. **Demo only, not for clinical use.**"
    )

single_tab, batch_tab = st.tabs(["Single patient", "Batch prediction"])

# --- Tab 1: one patient, entered as raw text ------------------------------------------------------
with single_tab:
    st.write(
        "Enter values as they appear in the raw records; messy formats such as `twenty`, `22kg/m2` or "
        "`120 over 80` are cleaned by the same rules as the training pipeline. Leave a field blank to "
        "have it filled with the value learned during preprocessing."
    )
    st.button("Fill with a random example", on_click=fill_random_example)

    with st.form("single_patient"):
        columns = st.columns(3)
        for i, col in enumerate(FEATURES):
            columns[i % 3].text_input(col, key=f"field_{col}")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw = pd.DataFrame([{"Patient_ID": "DEMO-1", **{col: st.session_state[f"field_{col}"] for col in FEATURES}}])
        raw = as_pipeline_input(raw)
        result = predict(raw, cfg, artifacts).iloc[0]

        label_col, prob_col = st.columns(2)
        label_col.metric("Prediction", result["predicted_label"].title())
        prob_col.metric("Probability of disease", f"{result['probability_disease']:.1%}")
        st.progress(float(result["probability_disease"]))

        st.subheader("How the input was cleaned")
        unfilled = clean_columns(raw[FEATURES])
        cleaned = preprocess_for_inference(raw, artifacts["fill_values"])
        st.dataframe(
            pd.DataFrame({
                "Raw input": [st.session_state[f"field_{col}"] for col in FEATURES],
                "Cleaned value": [str(cleaned.iloc[0][col]) for col in FEATURES],
                "Filled with training value": [bool(pd.isna(unfilled.iloc[0][col])) for col in FEATURES],
            }, index=FEATURES),
            width="stretch",
        )

# --- Tab 2: a raw CSV of many patients ------------------------------------------------------------
with batch_tab:
    source = st.radio("Input", ["Use the synthetic test data", "Upload a raw CSV"], horizontal=True)

    batch = None
    if source == "Upload a raw CSV":
        uploaded = st.file_uploader(
            "CSV with the raw columns (Patient_ID is optional; a Has_Disease column is ignored)", type="csv"
        )
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
    else:
        batch = pd.read_csv(project_path(cfg["inference"]["synthetic_path"]))

    if batch is not None:
        if "Patient_ID" not in batch.columns:
            batch.insert(0, "Patient_ID", [f"ROW-{i}" for i in range(1, len(batch) + 1)])
        try:
            predictions = predict(batch, cfg, artifacts)
        except ValueError as error:
            st.error(f"Could not score this file: {error}")
        else:
            total, flagged, mean_prob = st.columns(3)
            total.metric("Patients", len(predictions))
            flagged.metric("Predicted disease", int(predictions["predicted_class"].sum()))
            mean_prob.metric("Mean probability of disease", f"{predictions['probability_disease'].mean():.1%}")

            st.dataframe(predictions, width="stretch", hide_index=True)
            st.download_button(
                "Download predictions (CSV)",
                predictions.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )
