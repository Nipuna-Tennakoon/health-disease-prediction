# Reports

## 1.1 Data quality overview

![Data quality overview](figures/01_data_quality_overview.png)

### Key interpretation

- **Age and Heart_Rate are the least usable**: only ~26% valid; the rest are impossible sentinels (`-5`, `250`, `0`, `500`) or words. These must be treated as missing, not clipped as outliers.
- **Cholesterol_Level mixes numbers and labels**: 39% are `normal`/`high`, so it is better modelled as a category than a number.
- **Blood_Pressure and BMI look clean but are dominated by defaults**: `120/80` (~60%) and `22.5` (~51%) are in range but likely filler, so their real variance is small.
- **Every column needs cleaning**: no feature is more than 90% valid, and categorical columns are 10–40% missing. Imputation and missing-value handling are central to preprocessing.

### A. Numeric measurements

| Legend | Meaning | Examples |
|---|---|---|
| valid | Parseable number within the accepted range. Unit suffixes and alternate formats count as valid. | `45`, `22kg/m2`, `120 over 80` |
| invalid | Parseable number outside the accepted range, or systolic BP not above diastolic. | Age `250`, Heart_Rate `500`, BP `90/95` |
| null | Empty or placeholder value. | blank, `N/A`, `Unknown` |
| unparseable text | Text instead of a number, including numbers written as words. | `twenty`, `eighty`, `high`, `two weeks` |

Accepted ranges: Age 0–120 (whole numbers), BMI 10–70, Heart_Rate 30–220, Blood_Pressure systolic 70–250 / diastolic 40–150, Cholesterol_Level 100–400, Follow_Up ≥ 1 day.

### B. Categorical columns

| Legend | Meaning | Examples |
|---|---|---|
| valid | Any value that maps to a known category, including spelling and case variants. | `M`, `FEMALE`, `Nwe Yrok`, `None` (no medication) |
| null | Empty or placeholder value. | blank, `Unknown`, `n/a` |

### Excluded columns

- **Patient_ID**: identifier, not a feature; 507 rows (5%) are blank.
- **Name**: identifier with only 5 distinct values.
- **Last_Visit_Date**: all values parse (4 formats, slash dates are dd/mm/yyyy), ranging 2020-01-01 to 2025-10-01.

## 1.2 Target distribution

![Target distribution](figures/02_target_distribution.png)

### Key interpretation

- **Half the data is unlabelled**: 50% of rows are `unknown` or blank and must be excluded from training and evaluation, leaving 4,974 rows.
- **Labelled classes are balanced** (51% / 49%): no resampling or class weighting is needed, and accuracy is a fair metric alongside ROC-AUC and F1.
- **Baseline to beat is ~51%**: a model always predicting `0` scores this, so any useful model must clearly exceed it.

| Label | Rows | Share |
|---|---|---|
| `0` | 2,513 | 25% |
| `1` | 2,461 | 25% |
| `unknown` | 2,506 | 25% |
| blank | 2,520 | 25% |

## 1.3 Leakage check

![Leakage check](figures/03_leakage_check.png)

### Key interpretation

- **No leakage found**: every `Diagnosis_Code` and `Notes` value has a disease rate of 47–51%, within the 95% confidence interval of the overall 49%.
- **Both columns should still be dropped**: they carry no signal here, and in a real deployment they are recorded after diagnosis, so they would not be available at prediction time.

## 1.4 Disease rate by category

![Disease rate by category](figures/04_disease_rate_by_category.png)

### Key interpretation

- **No category separates the classes**: every group sits at 48–51%, within the confidence interval of the overall 49%.
- **Missingness is not informative**: Missing groups have the same disease rate, so missing-value indicator features are unlikely to help.
- **Clinically expected risk factors show no effect**: smokers, diabetics and high-cholesterol patients are no more likely to have the disease, suggesting the target is close to random relative to these features (likely synthetic data).
- **Expect near-chance model performance**: results should be judged against the ~51% baseline, not against typical clinical benchmarks.

Raw variants were mapped to one label for plotting only (e.g. `NY`, `newyork`, `Nwe Yrok` → New York; `190`/`normal` → Normal; `EX-smoker`/`Former` → Former). Blanks and placeholders are shown as **Missing**.
