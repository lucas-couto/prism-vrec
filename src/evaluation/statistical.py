"""Statistical methods for comparing recommendation models.

The functions in this module operate on per-user metric scores
produced by :class:`~src.evaluation.protocol.Evaluator`.  They are
deliberately stateless and dataframe-friendly so they can be combined
into different reporting workflows from :mod:`src.steps.statistical`.

Provided methods
----------------

* **Wilcoxon signed-rank** (``wilcoxon_test`` / ``pairwise_significance``)
  — non-parametric paired test, the workhorse for two-model comparisons.

* **Friedman test** (``friedman_test``) — non-parametric omnibus test
  for *N* models on the same users.  Use this first to check whether
  *any* model differs from the rest before running pairwise tests.

* **Multiple-comparison corrections** (``bonferroni_correction``,
  ``holm_bonferroni``) — control the family-wise error rate when many
  pairwise tests are reported together.  Holm is uniformly more
  powerful than vanilla Bonferroni.

* **Effect sizes** (``cohens_d_paired``, ``cliffs_delta``) — magnitude
  of the difference, complementary to p-values.  Always report both.

* **Bootstrap confidence intervals** (``bootstrap_ci``,
  ``per_model_summary``) — non-parametric estimate of the metric mean
  uncertainty.  Robust against non-normal score distributions.

Notes on cross-validation
-------------------------

The pipeline uses the standard recommender-systems protocol: a single
leave-one-out split with full-ranking evaluation against the test set.
A *full* K-fold cross-validation that retrains every recommender K
times is not implemented because it multiplies the cost of a 10K+
grid-search run by K with no expected change in the relative ranking
of models — the standard practice in this literature is to defend
robustness via paired non-parametric tests + bootstrap confidence
intervals on the held-out set, which is exactly what this module
exposes.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Returns ``(0.0, 1.0)`` when the two arrays are identical (the test
    is undefined if every paired difference is zero).
    """
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return 0.0, 1.0
    stat, p_value = wilcoxon(scores_a, scores_b, alternative="two-sided")
    return float(stat), float(p_value)


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Multiply each p-value by ``len(p_values)`` (capped at 1.0).

    Returns ``(corrected_p, is_significant)`` per input.
    """
    m = len(p_values)
    if m == 0:
        return []
    return [(min(p * m, 1.0), min(p * m, 1.0) < alpha) for p in p_values]


def holm_bonferroni(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Step-down Holm-Bonferroni correction.

    Sorts the p-values ascending and multiplies the i-th one by
    ``m - i`` (i.e. the remaining number of tests at that step), then
    enforces monotonicity so a later corrected value is never smaller
    than an earlier one.  Uniformly more powerful than vanilla
    Bonferroni while controlling the same family-wise error rate.

    Returns ``(corrected_p, is_significant)`` per input, in the
    original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values, dtype=float)[order]
    corrected_sorted = np.empty(m, dtype=float)

    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adjusted = min(p * (m - i), 1.0)
        running_max = max(running_max, adjusted)  # enforce monotonicity
        corrected_sorted[i] = running_max

    corrected = np.empty(m, dtype=float)
    corrected[order] = corrected_sorted
    return [(float(c), bool(c < alpha)) for c in corrected]


def cohens_d_paired(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Cohen's d for paired samples: ``mean(diff) / std(diff)``.

    Conventional thresholds: ``|d| < 0.2`` small, ``< 0.5`` medium,
    ``≥ 0.8`` large.  Returns ``0.0`` when the differences have zero
    variance (everyone changed by the same amount).
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def cliffs_delta(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Cliff's delta non-parametric effect size in ``[-1, 1]``.

    ``δ = P(A > B) - P(A < B)``.  Conventional thresholds (Romano et
    al. 2006): ``|δ| < 0.147`` negligible, ``< 0.33`` small, ``< 0.474``
    medium, ``≥ 0.474`` large.

    Implemented in O(n log n) via merge-sort rank counting so it scales
    to large per-user score arrays without a quadratic blow-up.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    n_a = len(a)
    n_b = len(b)
    if n_a == 0 or n_b == 0:
        return 0.0

    sorted_b = np.sort(b)
    greater = np.searchsorted(sorted_b, a, side="left").sum()
    less_or_equal = np.searchsorted(sorted_b, a, side="right").sum()
    n_greater = greater
    n_less = n_a * n_b - less_or_equal
    delta = (n_greater - n_less) / (n_a * n_b)
    return float(delta)


def cliffs_delta_magnitude(delta: float) -> str:
    """Map ``cliffs_delta`` to a textual magnitude (negligible/small/medium/large)."""
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    if abs_d < 0.33:
        return "small"
    if abs_d < 0.474:
        return "medium"
    return "large"


def bootstrap_ci(
    scores: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval for the mean.

    Returns ``(point_estimate, lower, upper)`` for the mean of
    ``scores``.  Confidence level is ``1 - alpha``.

    Resamples with replacement ``n_iterations`` times; uses the
    percentile method (no bias-corrected acceleration — fine for
    well-behaved metric distributions on hundreds-of-thousands of
    users).
    """
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    means = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()

    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lower, upper


def _ensure_config(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return *results_df* with a unique-per-(user, config) ``config`` column.

    The project-wide cell identity is ``model_name + "_" + embedding_name``
    (the same key the aggregated path uses, see ``steps/statistical.py``).
    A single ``model_name`` spans many embeddings, so grouping/pivoting on
    ``model_name`` alone collapses distinct cells and breaks paired tests.

    Non-visual baselines (``bpr`` / ``none``) are written to both battery
    files; in ``condition="all"`` they therefore appear twice per user
    with identical metrics. Deduplicating on ``(user_id, config)`` keeps
    one — lossless, and required so the pivots have a unique index.
    """
    out = results_df.copy()
    if "embedding_name" in out.columns:
        out["config"] = out["model_name"].astype(str) + "_" + out["embedding_name"].astype(str)
    else:
        out["config"] = out["model_name"].astype(str)
    if "user_id" in out.columns:
        out = out.drop_duplicates(subset=["user_id", "config"])
    return out


def per_model_summary(
    results_df: pd.DataFrame,
    metric: str,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per config with ``mean``, ``ci_lower/upper``, ``n_users``.

    A *config* is ``model_name + "_" + embedding_name``. Expects a
    long-format DataFrame with ``user_id``, ``model_name``,
    ``embedding_name`` and the requested ``metric`` column.
    """
    if metric not in results_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame columns: {list(results_df.columns)}"
        )

    df = _ensure_config(results_df)
    rows: list[dict] = []
    for config, group in df.groupby("config", sort=True):
        scores = group[metric].dropna().to_numpy()
        mean, lo, hi = bootstrap_ci(
            scores,
            n_iterations=n_iterations,
            alpha=alpha,
            seed=seed,
        )
        rows.append(
            {
                "config": config,
                "n_users": int(len(scores)),
                "mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_width": hi - lo,
            }
        )
    return pd.DataFrame(rows)


def friedman_test(
    results_df: pd.DataFrame,
    metric: str,
    alpha: float = 0.05,
) -> dict:
    """Friedman test across all configs on the same set of users.

    A *config* is ``model_name + "_" + embedding_name``. Returns a dict
    with ``statistic``, ``p_value``, ``significant`` (bool against
    ``alpha``), ``n_configs``, ``n_users``.

    Use this as a *gate* before pairwise testing — only run pairwise
    Wilcoxon when ``significant`` is True (otherwise none of the
    pairwise effects can be claimed at the family level).
    """
    if metric not in results_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame columns: {list(results_df.columns)}"
        )

    df = _ensure_config(results_df)
    pivot = df.pivot(index="user_id", columns="config", values=metric).dropna()
    if pivot.shape[1] < 3:
        # Friedman undefined for < 3 groups; the pairwise test is the right tool.
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "n_configs": int(pivot.shape[1]),
            "n_users": int(pivot.shape[0]),
            "note": "friedman undefined for fewer than 3 configs; use pairwise Wilcoxon instead",
        }

    columns = [pivot[col].to_numpy() for col in pivot.columns]
    stat, p_value = friedmanchisquare(*columns)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "n_configs": int(pivot.shape[1]),
        "n_users": int(pivot.shape[0]),
    }


_VALID_CORRECTIONS = {"bonferroni", "holm", "none"}


def pairwise_significance(
    results_df: pd.DataFrame,
    metric: str = "ndcg@10",
    alpha: float = 0.05,
    correction: str = "holm",
    include_effect_size: bool = True,
) -> pd.DataFrame:
    """All-pairs Wilcoxon test with multiple-comparison correction and effect sizes.

    Parameters
    ----------
    results_df:
        Long-format DataFrame with columns
        ``[user_id, model_name, embedding_name, <metric>]``.
    metric:
        Name of the metric column to compare.
    alpha:
        Family-wise significance level after correction.
    correction:
        ``"holm"`` (default — uniformly more powerful than Bonferroni),
        ``"bonferroni"``, or ``"none"``.
    include_effect_size:
        When True, adds Cohen's d and Cliff's delta columns.

    Returns
    -------
    pd.DataFrame
        One row per config pair with columns ``config_a``, ``config_b``,
        ``mean_a``, ``mean_b``, ``statistic``, ``p_value``,
        ``corrected_p``, ``significant``, plus (optionally)
        ``cohens_d``, ``cliffs_delta``, ``cliffs_magnitude``.
    """
    if correction not in _VALID_CORRECTIONS:
        raise ValueError(
            f"correction must be one of {sorted(_VALID_CORRECTIONS)}; got {correction!r}"
        )
    if metric not in results_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame columns: {list(results_df.columns)}"
        )

    df = _ensure_config(results_df)
    configs = sorted(df["config"].unique())
    if len(configs) < 2:
        raise ValueError(
            f"Need at least 2 distinct configs for pairwise comparison, got {len(configs)}."
        )

    pivot = df.pivot(index="user_id", columns="config", values=metric)

    rows: list[dict] = []
    raw_p_values: list[float] = []

    for config_a, config_b in combinations(configs, 2):
        valid = pivot[[config_a, config_b]].dropna()
        scores_a = valid[config_a].to_numpy()
        scores_b = valid[config_b].to_numpy()

        stat, p_val = wilcoxon_test(scores_a, scores_b)
        raw_p_values.append(p_val)

        row = {
            "config_a": config_a,
            "config_b": config_b,
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "statistic": stat,
            "p_value": p_val,
        }
        if include_effect_size:
            d = cohens_d_paired(scores_a, scores_b)
            delta = cliffs_delta(scores_a, scores_b)
            row["cohens_d"] = d
            row["cliffs_delta"] = delta
            row["cliffs_magnitude"] = cliffs_delta_magnitude(delta)
        rows.append(row)

    if correction == "bonferroni":
        corrected = bonferroni_correction(raw_p_values, alpha=alpha)
    elif correction == "holm":
        corrected = holm_bonferroni(raw_p_values, alpha=alpha)
    else:  # "none"
        corrected = [(p, p < alpha) for p in raw_p_values]

    for row, (corrected_p, is_sig) in zip(rows, corrected, strict=False):
        row["corrected_p"] = corrected_p
        row["significant"] = is_sig

    return pd.DataFrame(rows)
