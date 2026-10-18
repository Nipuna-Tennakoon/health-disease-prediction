"""Statistical comparison of models on repeated cross-validation scores.

Models are compared on the same folds with the Nadeau & Bengio corrected resampled t-test,
p-values are Holm-adjusted for the number of pairs, and the selection rule only declares a
winner when the evidence supports it.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def corrected_resampled_ttest(scores_a, scores_b, test_train_ratio: float) -> tuple[float, float, float]:
    """Paired t-test on per-fold scores, corrected for overlapping training sets (Nadeau & Bengio, 2003).

    CV folds share most of their training rows, so their scores are correlated and the ordinary
    paired t-test overstates significance. The correction inflates the variance by the held-out to
    training size ratio (1/4 for 5-fold). Returns (mean difference a - b, t statistic, two-sided p).
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n = len(diff)
    mean = float(diff.mean())
    variance = float(diff.var(ddof=1))
    if variance == 0:
        # Identical difference on every fold (e.g. two constant baselines): no evidence either way unless it is non-zero
        return mean, 0.0 if mean == 0 else float(np.copysign(np.inf, mean)), 1.0 if mean == 0 else 0.0
    t_stat = mean / np.sqrt((1 / n + test_train_ratio) * variance)
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    return mean, float(t_stat), float(p_value)


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjustment: keeps the chance of any false 'significant' result at alpha across all pairs."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running_max = max(running_max, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running_max
    return adjusted


def pairwise_comparison(fold_scores: dict[str, list[float]], test_train_ratio: float, alpha: float) -> pd.DataFrame:
    """Corrected t-test for every pair of models (baseline included), with Holm-adjusted p-values."""
    rows = []
    for model_a, model_b in itertools.combinations(fold_scores, 2):
        mean_diff, t_stat, p_value = corrected_resampled_ttest(fold_scores[model_a], fold_scores[model_b], test_train_ratio)
        rows.append({"model_a": model_a, "model_b": model_b, "mean_diff": mean_diff, "t_stat": t_stat, "p_value": p_value})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_value"])
    table["significant"] = table["p_holm"] < alpha
    return table


def _compare(significance: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, bool]:
    """Mean difference (a - b) and significance flag for one pair, whichever order it was stored in."""
    row = significance[(significance["model_a"] == model_a) & (significance["model_b"] == model_b)]
    if not row.empty:
        return float(row["mean_diff"].iloc[0]), bool(row["significant"].iloc[0])
    row = significance[(significance["model_a"] == model_b) & (significance["model_b"] == model_a)]
    return -float(row["mean_diff"].iloc[0]), bool(row["significant"].iloc[0])


def select_model(means: dict[str, float], significance: pd.DataFrame, baseline: str, simplicity_order: list[str]) -> dict:
    """Pick the best model only as far as the statistics allow.

    1. Eligible models beat the baseline significantly; if none do, every model stays in the pool.
    2. Models not significantly different from the top model (by mean) are treated as tied with it.
    3. Among tied models, the simplest one wins (simplicity_order), favouring explainable, stable models.
    """
    models = [m for m in means if m != baseline]
    beats_baseline = []
    for model in models:
        diff, significant = _compare(significance, model, baseline)
        if diff > 0 and significant:
            beats_baseline.append(model)
    pool = beats_baseline or models

    top = max(pool, key=means.get)
    tied = [m for m in pool if m == top or not _compare(significance, m, top)[1]]

    def simplicity_rank(model: str) -> tuple[int, float]:
        rank = simplicity_order.index(model) if model in simplicity_order else len(simplicity_order)
        return rank, -means[model]

    return {
        "name": min(tied, key=simplicity_rank),
        "significant": bool(beats_baseline),
        "top_by_mean": top,
        "tied_with_top": tied,
        "beats_baseline": beats_baseline,
    }
