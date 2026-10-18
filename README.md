# Health Disease Prediction

## 1. Project overview

This project builds a reproducible machine learning pipeline that predicts whether a patient has a disease (`Has_Disease`) from raw patient health records. It covers data-quality analysis, rule-based cleaning, leakage-safe encoding and scaling, a statistically tested comparison of four classifiers (repeated 5×5 cross-validation against a no-skill baseline), and an inference workflow that scores new raw records. Experiments are tracked in Weights & Biases. The same logic is packaged as a Python library with a CLI and three Airflow DAGs, so the whole pipeline can run automatically.

## 2. Problem statement

- **Objective:** binary classification of `Has_Disease` (1 = disease, 0 = no disease).
- **Output:** for each patient, a predicted class and a probability of disease.
- **Data:** [Patient Health Record – Raw CSV (Kaggle)](https://www.kaggle.com/datasets/mursaleen880/patient-health-record-raw-csv), 10,000 rows × 17 columns.
- **Which file:** the raw file is used rather than the provided "clean" file. That file was scaled and encoded on the full dataset before any split, which leaks test information. Cleaning from raw keeps every transformation under our control.

## 3. Repository structure

```
health-disease-prediction/
├── README.md
├── config.yml                  # data paths, features, model settings and versions, W&B, best model
├── pyproject.toml / uv.lock    # dependencies (uv)
├── .env.example                # W&B credentials template (copy to .env)
├── notebooks/
│   ├── 01_visualization.ipynb  # EDA, data quality, 4 figures
│   ├── 02_preprocessing.ipynb  # cleaning rules -> cleaned dataset
│   ├── 03_training.ipynb       # split, encoding, repeated CV, 4 models, W&B, statistical comparison, selection
│   └── 04_inference.ipynb      # synthetic records -> predictions
├── src/health_disease_prediction/   # the notebooks' logic as a reusable package
│   ├── preprocessing.py  features.py  training.py  evaluation.py  inference.py
│   ├── config.py  cli.py  orchestration.py
├── dags/                       # Airflow DAGs: preprocessing, training, inference
├── tests/                      # pytest unit tests
├── data/                       # raw, processed, synthetic data (not committed, see data/README.md)
├── artifacts/                  # versioned encoders, fill values and models (*_v1.*)
└── reports/
    ├── figures/                # EDA figures
    ├── model_comparison.csv    # mean ± std per model (CV), train and test metrics
    ├── cv_fold_scores.csv      # every model's score on each of the 25 CV folds
    ├── model_significance.csv  # pairwise corrected t-tests with Holm-adjusted p-values
    ├── synthetic_predictions.csv
    └── README.md               # full figure legends and interpretations
```

## 4. Methodology

### 4.1 Exploratory data analysis
Each column was classified as **valid / invalid / null / unparseable text** against clinical ranges. The analysis also covered the target distribution, a leakage check and the disease rate by category with 95% confidence intervals (see section 5).

### 4.2 Preprocessing
Rows without a confirmed 0/1 label are dropped, leaving **4,974 of 10,000** rows. Each column is then cleaned with explicit rules:

| Column | Cleaning rule | Missing filled with |
|---|---|---|
| Age | words to numbers (`twenty` → 20); values > 120 divided by 10 (`250` → 25); negatives → missing | mode |
| Gender | `male`/`M` → M, `FEMALE`/`F` → F | mode |
| City | merge variants (`NY`, `newyork`, `Nwe Yrok` → New York; `la` → Los Angeles) | mode |
| BMI | strip `kg/m2` | mean |
| Blood_Pressure | `120 over 80`, `120 - 80` → `120/80` | mode |
| Heart_Rate | keep 40–180 bpm; `0`, `500`, words → missing | median |
| Cholesterol_Level | `high` → 250, `normal` → 125, `low` → 100 | mode |
| Diabetic / Smoker | yes/no variants → `Y`/`N`; smoker `EX-smoker`/`Former` → `Ex` | mode |
| Medications | unify separators (`;`, `/` → `,`) | `none` |
| Follow_Up | days as numbers; `two weeks` → missing | mode |
| Diagnosis_Code / Notes | placeholders → missing | `Unknown` / `No Notes` |
| Last_Visit_Date | four date formats parsed (slash dates are dd/mm/yyyy) | mode |

Patient_ID gaps are rebuilt from row order. Patient_ID, Name and Last_Visit_Date are not used as model features.

### 4.3 Splitting
A stratified 80/20 train/test split with a fixed seed (42) gives **3,979 training and 995 test rows**, with the same class balance in both.

### 4.4 Encoding and scaling
All encoders are fitted on the **training split only** and reused unchanged on test and inference data, producing 29 model features.

| Features | Method | Why |
|---|---|---|
| Gender, City, Medications, Diagnosis_Code, Notes, Smoker, Diabetic | One-hot | nominal categories; unseen values become all zeros |
| Blood_Pressure | Frequency encoding | ~700 distinct readings, too many for one-hot |
| Follow_Up | Ordinal (14 → 0, 30 → 1) | two ordered values |
| Age, BMI, Heart_Rate, Cholesterol_Level | Standard scaling | puts features on one scale for SVM and logistic regression |

### 4.5 Algorithms
Random Forest, Logistic Regression, SVM (RBF kernel) and XGBoost, plus a **no-skill baseline** (`DummyClassifier`, strategy `prior`). The baseline ignores every feature and gives each patient the training disease rate (49.5%) as their probability, so it scores ROC-AUC 0.5. Model parameters and versions (currently `1.0.0`) are set in `config.yml`.

### 4.6 Repeated cross-validation
- **Repeated stratified 5-fold cross-validation (5 folds × 5 repeats = 25 scores per model), on the training split only.** Each repeat reshuffles the training rows into new folds. The test set is never used here.
- **Identical folds for every model:** a fixed seed gives the models and the baseline exactly the same 25 folds, so their scores can be compared fold by fold.
- **Leakage-safe folds:** encoders and scaler are refitted inside every fold on that fold's training part, so each held-out fold is as unseen as new data.
- **Final model:** after cross-validation, each model is refitted on the full training split and scored **once** on the test set, for reporting only.
- **Tuning:** hyperparameters are set by hand in `config.yml`; no automated search has been run (see section 9).

### 4.7 Statistical comparison and model selection
Picking the highest average always produces a "winner", even when the differences are just noise. Instead, models are compared with significance tests:

1. **Corrected resampled t-test** (Nadeau & Bengio, 2003) for every pair: 4 models + baseline = 10 pairs, comparing their ROC-AUC on the same 25 folds. CV training sets overlap, which makes fold scores correlated, so the ordinary paired t-test overstates significance. The correction inflates the variance by the held-out-to-training size ratio:

   $$t = \frac{\bar d}{\sqrt{\left(\frac{1}{J} + \frac{n_{test}}{n_{train}}\right)\hat\sigma^2}}, \qquad J = 25,\ \frac{n_{test}}{n_{train}} = \frac{1}{4},\ df = 24$$

   where $\bar d$ and $\hat\sigma^2$ are the mean and variance of the 25 per-fold differences.
2. **Holm–Bonferroni correction** of the 10 p-values, so the chance of any false "significant" result stays at α = 0.05.
3. **Selection rule:**
   1. A model is eligible only if it beats the **baseline** significantly. If none does, all models stay in the pool.
   2. Models **not significantly different** from the top model (by mean ROC-AUC) are treated as tied with it.
   3. Among tied models, the **simplest** wins: Logistic Regression < Random Forest < XGBoost < SVM. This favours explainable, stable models when performance is equivalent.
   4. `config.yml` records the choice, with `significant: true` only if the model beats the baseline.

**Metrics:** accuracy, precision, recall, F1 and ROC-AUC, computed for every fold, the full training split and the test set. Comparing train against CV and test shows overfitting.

**Tracking (W&B):**
- **One run per model:** per-fold scores (a chart across the 25 splits and a table), the CV mean and std, train/test metrics and the test confusion matrix.
- **A `comparison-v1` run:** the significance table and all fold scores.

### 4.8 Inference
New raw records go through the same cleaning rules, are filled with the values saved during preprocessing, and are encoded with the saved encoders. The best model named in `config.yml` then scores them. As a demonstration, 25 synthetic records are generated by sampling the raw columns, so they keep the original messy formats.

### 4.9 Reproducibility and automation
- **Everything in one config:** paths, features, model settings and versions live in `config.yml`.
- **Versioned artifacts:** the files are `fill_values_v1.json`, `preprocessors_v1.joblib` and `models_v1.joblib`, and inference checks the model version against the config.
- **Fixed seeds:** all random steps use the same seed.
- **Experiment tracking:** every run is logged to W&B.
- **Automation:** the notebooks' logic is packaged in `src/` and runs either from the `hdp` command or from three Airflow DAGs linked by data assets: preprocessing → training → inference.

## 5. Visualizations and interpretations

Full legends are in [reports/README.md](reports/README.md).

### 5.1 Data quality overview
![Data quality overview](reports/figures/01_data_quality_overview.png)
- Age and Heart_Rate are only about 26% valid. The rest are impossible placeholder values (`-5`, `250`, `0`, `500`) or words.
- No column is more than 90% valid, so cleaning and imputation decide what the model sees.

### 5.2 Target distribution
![Target distribution](reports/figures/02_target_distribution.png)
- 50% of labels are `unknown` or blank and cannot be used for training.
- The labelled classes are balanced (51% / 49%), so no resampling is needed. The baseline to beat is about 51% accuracy.

### 5.3 Leakage check
![Leakage check](reports/figures/03_leakage_check.png)
- Every `Diagnosis_Code` and `Notes` value has a disease rate of 47–51%, so **no leakage** was found.
- In a real deployment these fields are recorded after diagnosis. They would not be available at prediction time, so they should be excluded.

### 5.4 Disease rate by category
![Disease rate by category](reports/figures/04_disease_rate_by_category.png)
- Every category, including missing values, sits at 48–51%, within the confidence interval of the overall 49%.
- Even clinically expected risk factors (smoking, diabetes, high cholesterol) show no effect, so expect near-chance model performance.

## 6. Results summary

### 6.1 Repeated cross-validation (training split, 5 × 5 folds, mean ± std)
| Model | CV accuracy | CV F1 | CV ROC-AUC | ROC-AUC range over 25 folds |
|---|---|---|---|---|
| XGBoost | 0.508 ± 0.015 | 0.498 ± 0.015 | **0.518 ± 0.016** | 0.496 – 0.548 |
| **Random Forest** (selected) | 0.494 ± 0.011 | 0.443 ± 0.025 | 0.498 ± 0.015 | 0.460 – 0.527 |
| SVM | 0.491 ± 0.015 | 0.476 ± 0.021 | 0.484 ± 0.015 | 0.455 – 0.516 |
| Logistic Regression | 0.487 ± 0.012 | 0.457 ± 0.012 | 0.483 ± 0.013 | 0.461 – 0.513 |
| *No-skill baseline* | *0.505* | *0.000* | *0.500* | *0.500 – 0.500* |

### 6.2 Statistical comparison (corrected resampled t-test on ROC-AUC, Holm-adjusted, α = 0.05)
| Model A | Model B | Mean difference (A − B) | t | p | p (Holm) | Significant |
|---|---|---|---|---|---|---|
| SVM | XGBoost | −0.034 | −3.98 | 0.001 | 0.006 | **Yes** |
| Logistic Regression | XGBoost | −0.035 | −3.05 | 0.006 | 0.050 | **Yes** |
| Logistic Regression | Baseline | −0.017 | −2.30 | 0.030 | 0.244 | No |
| Random Forest | XGBoost | −0.020 | −2.21 | 0.037 | 0.260 | No |
| XGBoost | Baseline | +0.018 | 2.04 | 0.052 | 0.313 | No |
| Random Forest | SVM | +0.015 | 1.99 | 0.058 | 0.313 | No |
| SVM | Baseline | −0.016 | −1.97 | 0.061 | 0.313 | No |
| Random Forest | Logistic Regression | +0.015 | 1.81 | 0.083 | 0.313 | No |
| Random Forest | Baseline | −0.002 | −0.20 | 0.845 | 1.000 | No |
| Logistic Regression | SVM | −0.000 | −0.04 | 0.965 | 1.000 | No |

Sources: [reports/model_significance.csv](reports/model_significance.csv) and [reports/cv_fold_scores.csv](reports/cv_fold_scores.csv).

### 6.3 Selected model
How the selection rule applied:

1. **No model beats the baseline significantly.** XGBoost comes closest (+0.018 ROC-AUC), but its p-value is 0.31 after the Holm correction. So all four models stay in the pool, and the result is recorded as `significant: false`.
2. **XGBoost has the highest mean** (0.518). Logistic Regression and SVM are significantly worse than it (Holm p = 0.050 and 0.006), so they are excluded. Random Forest is **not** significantly different (Holm p = 0.26), so it is tied with XGBoost.
3. **Simplest of the tied models:** Random Forest is simpler than XGBoost.

**Selected: Random Forest v1.0.0** (300 trees, max depth 5, at least 5 samples per leaf), with `significant: false`.

### 6.4 Holdout test set (n = 995, scored once after refitting on the full training split)
| Model | Train accuracy | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** (selected) | 0.640 | 0.489 | 0.480 | 0.390 | 0.431 | 0.495 |
| XGBoost | 0.774 | 0.498 | 0.492 | 0.480 | 0.486 | 0.495 |
| SVM | 0.673 | 0.503 | 0.497 | 0.494 | 0.495 | 0.492 |
| Logistic Regression | 0.515 | 0.486 | 0.478 | 0.423 | 0.449 | 0.484 |
| *No-skill baseline* | – | *0.506* | – | – | – | *0.500* |

Source: [reports/model_comparison.csv](reports/model_comparison.csv).

### 6.5 Confusion matrix (Random Forest, test set)
| | Predicted no disease | Predicted disease |
|---|---|---|
| **Actual no disease** (503) | 295 (true negative) | 208 (false positive) |
| **Actual disease** (492) | 300 (false negative) | 192 (true positive) |

### 6.6 Variability across folds
- **Spread:** ROC-AUC varies by ±0.013–0.016 (std) across the 25 folds. A single model's best and worst folds are 0.05–0.07 apart; Random Forest, for example, goes from 0.460 to 0.527.
- **Overlap:** every model's fold range includes 0.5, so on some folds each model is no better than the baseline.
- **CV ranking vs test ranking:** XGBoost leads in cross-validation (0.518) but ties Random Forest on the test set (0.495 each). Both are below 0.5. The test set confirms that the cross-validation lead was not real skill.

## 7. Interpretation of results

- **No model has learned anything useful.** None is significantly better than a baseline that ignores all features (all Holm-adjusted p ≥ 0.24 against the baseline). The selected Random Forest scores 48.9% test accuracy and 0.495 ROC-AUC, below always predicting "no disease" (50.6%, 0.500). This matches the EDA, where no feature separated the classes.
- **The statistical comparison changed the decision.** Picking the highest average would have crowned XGBoost. The tests show its lead over the baseline is not significant, and that it is statistically tied with Random Forest, so the simpler Random Forest is chosen. The only significant differences are XGBoost scoring higher than Logistic Regression and SVM, all within ±0.02 of 0.5, which isn't a clinically meaningful gap.
- **Overfitting:** XGBoost reaches 77% training accuracy but about 50% in CV and on test, so it memorises noise. The shallower Random Forest (depth 5) overfits less (64% train), and Logistic Regression's equal train and test scores show there's little signal to learn.
- **Key errors:** the selected model misses **61% of patients who have the disease** (300 of 492 are false negatives) and flags 41% of healthy patients (208 of 503 are false positives). For a disease model, missed cases are usually the most costly error.
- **Class trade-off:** lowering the decision threshold below 0.5 would catch more sick patients (higher recall) at the cost of more false alarms. With no underlying signal, this only moves errors between the two classes; it doesn't reduce them.
- **Deployment caution:** the model **must not be used for clinical decisions**, and `significant: false` in `config.yml` records why. Its synthetic-test probabilities stay between 0.45 and 0.55, an honest reflection that it can't tell patients apart.

## 8. Reproduction instructions

### 8.1 Environment
```bash
git clone <repo-url> && cd health-disease-prediction
uv sync                       # creates .venv with locked dependencies (Python 3.10)
cp .env.example .env          # add WANDB_API_KEY (and optionally WANDB_ENTITY)
```
To run without W&B, set `wandb.mode: disabled` in `config.yml`.

### 8.2 Option A: notebooks (exploration)
Open the notebooks with the `.venv` kernel and run them in order. Each step downloads or reads the previous step's outputs.

| Order | Notebook | Produces |
|---|---|---|
| 1 | `01_visualization.ipynb` | `data/Patient_Health_Records_Raw.csv`, `reports/figures/*.png` |
| 2 | `02_preprocessing.ipynb` | `data/processed/Cleaned_Health_Records.csv` |
| 3 | `03_training.ipynb` | `artifacts/encoders/`, `artifacts/models/`, `reports/model_comparison.csv`, `cv_fold_scores.csv`, `model_significance.csv`, best model in `config.yml` |
| 4 | `04_inference.ipynb` | `data/synthetic/Synthetic_Test_Data.csv`, predictions |

### 8.3 Option B: command line (same results as the notebooks)
```bash
uv run hdp preprocess    # download (if missing), clean, validate, save fill values
uv run hdp train         # split, fit encoders, 5x5 CV + train 4 models, significance tests vs baseline, select
uv run hdp predict       # generate synthetic records and save reports/synthetic_predictions.csv
uv run hdp all           # all three stages in sequence
```

### 8.4 Option C: Airflow
Trigger `health_preprocessing`; training and inference run automatically when their inputs update. Setup instructions are in [dags/README.md](dags/README.md).

### 8.5 Tests
```bash
uv run pytest
```

## 9. Assumptions and limitations

### Assumptions
- **Unlabelled rows:** rows with `unknown` or blank labels are assumed to be missing at random, so dropping them doesn't bias the training data.
- **Ages over 120** are assumed to be data-entry errors with an extra digit (`250` → 25).
- **Text cholesterol labels** are mapped to typical values: `normal` → 125, `high` → 250, `low` → 100 mg/dL.
- **Heart rates outside 40–180 bpm** are assumed to be recording errors.
- **Missing medication** means no medication (`none`).
- **Slash dates** are day-first (dd/mm/yyyy), because no value has a second part above 12.

### Limitations
- **Heavy imputation:** about 75% of Heart_Rate and about 50% of Follow_Up values are filled with a single value, so these features carry little information.
- **Fill values come from all labelled rows.** The mode, mean and median used to fill gaps are computed before the train/test split, so test rows have a small influence on them.
- **No automated hyperparameter tuning:** parameters are set by hand in `config.yml`. A search would need nested cross-validation to keep the comparison fair, and with no signal in the data it is unlikely to help.
- **The statistical tests are approximate.** The Nadeau–Bengio correction is a widely used heuristic, not an exact test. With 25 folds and Holm's correction, small real differences (around 0.01 ROC-AUC) may go undetected.
- **The data looks synthetic:** values repeat (e.g. `120/80` in about 60% of rows) and the target is unrelated to clinical risk factors. Results would not carry over to real patients.

