"""Clean the raw patient records into a model-ready dataset (same rules as 02_preprocessing.ipynb).

Cleaning is split into three steps so inference can reuse it:
  clean_columns       stateless parsing; unparseable values become NaN
  compute_fill_values statistics learned once, during preprocessing
  fill_missing        applies those saved statistics
"""

import io
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from health_disease_prediction.config import artifact_paths, project_path

logger = logging.getLogger(__name__)

# Placeholder tokens that mean "value not recorded"
EMPTY = ["", "nan", "none", "null", "n/a", "na", "unknown"]

AGE_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50}
GENDER_MAP = {"male": "M", "female": "F", "m": "M", "f": "F"}
CITY_MAP = {
    "boston": "Boston",
    "la": "Los Angeles",
    "losangeles": "Los Angeles",
    "newyork": "New York",
    "nweyrok": "New York",
    "ny": "New York",
    "chicago": "Chicago",
}
CHOLESTEROL_MAP = {"high": "250", "low": "100", "normal": "125"}
DIABETIC_MAP = {"yes": "Y", "y": "Y", "no": "N", "n": "N"}
SMOKER_MAP = {"yes": "Y", "y": "Y", "no": "N", "n": "N", "ex-smoker": "Ex", "former": "Ex"}
NAME_EMPTY = ["", "None", "Nan", "nan", "unknown", "n/a", "N/A"]


def clean_target(x):
    """Only confirmed 0/1 labels can be used for supervised training."""
    x = str(x).strip().lower()
    if x in ["1", "yes", "y"]:
        return 1
    if x in ["0", "no", "n"]:
        return 0
    return np.nan


def clean_age(x):
    """Ages are recorded as words, x10 entry errors (250) and negative sentinels (-5)."""
    x = str(x).strip().lower()
    if x in AGE_WORDS:
        return AGE_WORDS[x]
    try:
        val = float(x)
    except ValueError:
        return np.nan
    if val < 0:
        return np.nan
    return val / 10 if val > 120 else val


def clean_gender(x):
    """Collapse case and abbreviation variants into M/F."""
    x = str(x).lower().strip()
    x = GENDER_MAP.get(x, x)
    return np.nan if x in ["unknown", "nan", ""] else x


def clean_city(x):
    """Merge abbreviations and typos (NY, newyork, Nwe Yrok) into one city name."""
    x = str(x).lower().replace(" ", "").strip()
    if x in EMPTY:
        return np.nan
    return CITY_MAP.get(x, x.title())


def clean_bmi(x):
    """Strip the kg/m2 unit so every BMI is a plain number."""
    x = str(x).lower().strip()
    if x in EMPTY:
        return np.nan
    return float(x.replace("kg/m2", ""))


def clean_blood_pressure(x):
    """Unify '120 over 80' and '120 - 80' into the standard 'systolic/diastolic' format."""
    x = str(x).lower().strip()
    if x in EMPTY:
        return np.nan
    return x.replace(" over ", "/").replace(" - ", "/")


def clean_heart_rate(x):
    """Keep numeric rates in a realistic 40-180 bpm range; sentinels (0, 500) and words become missing."""
    x = str(x).strip().lower()
    if x in EMPTY or not x.replace(".", "").isdigit():
        return np.nan
    val = float(x)
    return val if 40 <= val <= 180 else np.nan


def clean_cholesterol(x):
    """Convert text levels (high/normal/low) to representative mg/dL values."""
    x = str(x).lower().strip()
    if x in EMPTY:
        return np.nan
    return CHOLESTEROL_MAP.get(x, x)


def clean_diabetic(x):
    """Map yes/no variants to Y/N."""
    x = str(x).strip().lower()
    if x in EMPTY:
        return np.nan
    return DIABETIC_MAP.get(x, np.nan)


def clean_smoker(x):
    """Smoking status: Y = current, N = never, Ex = former (EX-smoker/Former)."""
    x = str(x).strip().lower()
    if x in EMPTY:
        return np.nan
    return SMOKER_MAP.get(x, np.nan)


def clean_medications(x):
    """Unify list separators (',', ';', '/') so the same drug combination has one spelling."""
    x = str(x).lower().strip()
    if x in EMPTY:
        return np.nan
    x = x.replace(";", ",").replace("/", ",")
    return x.replace(" ,", ",").replace(", ", ",")


def clean_follow_up(x):
    """Keep follow-up as whole days; text such as 'two weeks' becomes missing."""
    x = str(x).strip().lower()
    if x in EMPTY:
        return np.nan
    try:
        return int(float(x))
    except ValueError:
        return np.nan


def clean_diagnosis_code(x):
    """Placeholder codes are missing."""
    return np.nan if x in EMPTY else x


def clean_notes(x):
    """Placeholder notes are missing."""
    x = str(x).strip()
    return np.nan if x in EMPTY else x


# Columns the model uses; inference cleans exactly these
FEATURE_CLEANERS = {
    "Age": clean_age,
    "Gender": clean_gender,
    "City": clean_city,
    "BMI": clean_bmi,
    "Blood_Pressure": clean_blood_pressure,
    "Heart_Rate": clean_heart_rate,
    "Cholesterol_Level": clean_cholesterol,
    "Diabetic": clean_diabetic,
    "Smoker": clean_smoker,
    "Medications": clean_medications,
    "Follow_Up": clean_follow_up,
    "Diagnosis_Code": clean_diagnosis_code,
    "Notes": clean_notes,
}

CONSTANT_FILLS = {"Medications": "none", "Diagnosis_Code": "Unknown", "Notes": "No Notes"}
MEAN_FILLS = ["BMI"]
MEDIAN_FILLS = ["Heart_Rate"]


def clean_patient_id(s: pd.Series) -> pd.Series:
    """Raw IDs follow the row order (ID-00001 = row 1), so blank IDs are rebuilt from the original row position."""
    ids = pd.to_numeric(s.astype(str).str.replace("ID-", "", regex=False).str.strip(), errors="coerce")
    ids = ids.fillna(pd.Series(s.index + 1, index=s.index))
    return "ID-" + ids.astype(int).astype(str)


def clean_name(s: pd.Series) -> pd.Series:
    """Unify case and remove the 'Last, First' comma so the same person has one spelling."""
    s = s.str.title().str.replace(",", "").str.strip()
    return s.replace(NAME_EMPTY, np.nan)


def parse_visit_dates(s: pd.Series) -> pd.Series:
    """Dates come in four formats; parse each value individually (slash dates are dd/mm/yyyy)."""
    return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse every feature column; values that cannot be parsed are left as NaN for fill_missing."""
    out = df.copy()
    for col, cleaner in FEATURE_CLEANERS.items():
        out[col] = df[col].apply(cleaner)
    return out


def compute_fill_values(df: pd.DataFrame) -> dict:
    """Learn one fill value per feature column from the cleaned (still unfilled) data."""
    fill_values = {}
    for col in FEATURE_CLEANERS:
        if col in CONSTANT_FILLS:
            fill_values[col] = CONSTANT_FILLS[col]
        elif col in MEAN_FILLS:
            fill_values[col] = df[col].mean()
        elif col in MEDIAN_FILLS:
            fill_values[col] = df[col].median()
        else:
            fill_values[col] = df[col].mode()[0]
    return fill_values


def fill_missing(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    """Fill feature gaps with the saved values and restore the dtypes of the cleaned training data."""
    out = df.copy()
    for col, value in fill_values.items():
        out[col] = out[col].fillna(value)
    out["Cholesterol_Level"] = out["Cholesterol_Level"].astype(float)
    out["Follow_Up"] = out["Follow_Up"].astype(float)
    return out


def _to_json_safe(fill_values: dict) -> dict:
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in fill_values.items()}


def save_fill_values(fill_values: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_safe(fill_values), indent=2))


def load_fill_values(path: Path) -> dict:
    return json.loads(path.read_text())


def download_raw_data(cfg: dict) -> Path:
    """Fetch the raw CSV from Kaggle unless it is already present."""
    raw_path = project_path(cfg["data"]["raw_path"])
    if raw_path.exists():
        logger.info("Raw data already present at %s", raw_path)
        return raw_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(cfg["data"]["raw_url"]) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extract(raw_path.name, raw_path.parent)
    logger.info("Downloaded raw data to %s", raw_path)
    return raw_path


def run_preprocessing(cfg: dict) -> Path:
    """Raw CSV -> cleaned CSV, plus the fill values inference needs."""
    target = cfg["data"]["target"]
    df = pd.read_csv(project_path(cfg["data"]["raw_path"]))

    # Unknown and blank labels cannot be trained on; the original index is kept for Patient_ID recovery
    df[target] = df[target].apply(clean_target)
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    df = clean_columns(df)
    fill_values = compute_fill_values(df)
    df = fill_missing(df, fill_values)

    dates = parse_visit_dates(df["Last_Visit_Date"])
    df["Last_Visit_Date"] = dates.fillna(dates.mode()[0])
    df["Patient_ID"] = clean_patient_id(df["Patient_ID"])
    names = clean_name(df["Name"])
    df["Name"] = names.fillna(names.mode()[0])

    cleaned_path = project_path(cfg["data"]["cleaned_path"])
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    save_fill_values(fill_values, artifact_paths(cfg)["fill_values"])
    logger.info("Saved cleaned dataset %s to %s", df.shape, cleaned_path)
    return cleaned_path


def validate_cleaned_data(cfg: dict) -> dict:
    """Fail the pipeline early if the cleaned dataset is not model-ready."""
    df = pd.read_csv(project_path(cfg["data"]["cleaned_path"]))
    target = cfg["data"]["target"]
    expected = set(FEATURE_CLEANERS) | set(cfg["data"]["drop_columns"]) | {target}

    missing_columns = expected - set(df.columns)
    if missing_columns:
        raise ValueError(f"Cleaned data is missing columns: {sorted(missing_columns)}")
    if df.isna().any().any():
        raise ValueError(f"Cleaned data has missing values: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")
    if not set(df[target].unique()) <= {0, 1}:
        raise ValueError(f"Target has values other than 0/1: {sorted(df[target].unique())}")

    summary = {"rows": len(df), "columns": df.shape[1], "positive_rate": round(float(df[target].mean()), 4)}
    logger.info("Cleaned data is valid: %s", summary)
    return summary
