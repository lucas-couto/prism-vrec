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
  powerful than vanilla Bonferroni.  The correction is applied WITHIN
  a comparison family (:mod:`src.evaluation.comparison_families`) —
  the set of hypotheses one research question defines — never over the
  Cartesian product of every config.

* **Effect sizes** — **Cliff's delta** (``cliffs_delta``) is the
  primary one: non-parametric and tie-robust, consistent with
  Wilcoxon+pratt on 0/1-heavy LOO metrics.  ``cohens_d_paired`` is
  parametric and inflates on zero-dominated differences; it is kept
  for diagnostics only (off by default in reported tables).

* **Bootstrap confidence intervals** (``bootstrap_ci``,
  ``per_model_summary``, ``bootstrap_diff_ci``) — non-parametric
  uncertainty for the per-config mean (descriptive) and for the
  PAIRED mean difference (inferential; the CI that must agree with
  the Wilcoxon verdict).

Metric redundancy under leave-one-out
-------------------------------------

With exactly one relevant item per user, recall@k is 0/1 (HitRate@k),
precision@k = recall@k / k, map@k = 1/rank of the single hit, and
ndcg@k = 1/log2(rank+1) — only two independent signals exist (hit
or not; at which rank).  The reporting step analyses recall + ndcg as
primary and treats precision/map as derived, never as independent
evidence.

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

    Uses ``zero_method="pratt"`` so zero differences are kept in the
    ranking rather than discarded.  With per-user leave-one-out metrics
    (recall@k is 0/1 per user) the vast majority of paired differences
    are exactly zero; scipy's default (``"wilcox"``) drops them all,
    shrinking the effective sample far below ``n_users`` and inflating
    apparent effects on the tiny nonzero remainder.  Pratt's treatment
    keeps the zeros in the rank computation, which is the conservative
    choice for tie-heavy paired metrics.

    Returns ``(0.0, 1.0)`` when the two arrays are identical (the test
    is undefined if every paired difference is zero).
    """
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return 0.0, 1.0
    stat, p_value = wilcoxon(scores_a, scores_b, alternative="two-sided", zero_method="pratt")
    return float(stat), float(p_value)


def n_nonzero_pairs(scores_a: np.ndarray, scores_b: np.ndarray) -> int:
    """Number of paired differences that are not exactly zero.

    Reported alongside the Wilcoxon p-value so the reader can see how
    much of ``n_users`` actually carries signal in tie-heavy metrics.
    """
    return int(np.count_nonzero(np.asarray(scores_a) - np.asarray(scores_b)))


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

    .. warning::
        **Secondary/diagnostic only under this pipeline's LOO metrics.**
        Cohen's d is parametric (assumes roughly normal differences);
        per-user LOO metrics are 0/1-heavy, so the difference vector is
        dominated by zeros with a few ``±1`` — the ``std`` shrinks and
        ``d`` inflates without a sensible interpretation.  The primary
        effect size is :func:`cliffs_delta` (non-parametric, tie-robust),
        consistent with the choice of Wilcoxon + ``pratt``.  This column
        is therefore OFF by default in the reported tables
        (``statistical.include_cohens_d``).

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


def bootstrap_diff_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI of the PAIRED mean difference ``mean(A - B)``.

    Resamples USERS (paired rows), not configs, so the between-user
    variance that the paired Wilcoxon removes is also removed here.
    Individual per-config CIs (:func:`bootstrap_ci`) can overlap while
    the paired test is highly significant — that is not a
    contradiction, it is unpaired vs paired variance.  This CI is the
    one that must agree with the Wilcoxon verdict: a paired-difference
    CI excluding zero is consistent with a significant paired test.

    Returns ``(diff_mean, lower, upper)``.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    return bootstrap_ci(diff, n_iterations=n_iterations, alpha=alpha, seed=seed)


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
    configs: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Friedman test across configs on the same set of users.

    A *config* is ``model_name + "_" + embedding_name``.  When *configs*
    is given, the omnibus runs on that subset only — this is how the
    step applies it PER COMPARISON FAMILY (an omnibus over all ~77
    heterogeneous configs of a dataset answers no research question).
    Returns a dict with ``statistic``, ``p_value``, ``significant``
    (bool against ``alpha``), ``n_configs``, ``n_users``.

    Use this as a *gate* before pairwise testing — only run pairwise
    Wilcoxon when ``significant`` is True (otherwise none of the
    pairwise effects can be claimed at the family level).
    """
    if metric not in results_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame columns: {list(results_df.columns)}"
        )

    df = _ensure_config(results_df)
    if configs is not None:
        df = df[df["config"].isin(set(configs))]
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
    pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
    family: str = "all_pairs",
    group: str = "all",
    include_cohens_d: bool = False,
    diff_ci: bool = True,
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Pairwise Wilcoxon tests with correction applied WITHIN one family.

    The correction's ``m`` is the number of *pairs tested in this call*
    — pass the pairs of ONE comparison family (see
    :mod:`src.evaluation.comparison_families`) so Holm corrects within
    the set of hypotheses the research question actually defines.
    Called without *pairs*, it degrades to the exploratory all-pairs
    mode over every config in *results_df*.

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
        When True, adds the Cliff's delta columns (primary effect size).
    pairs / family / group:
        The comparison family: explicit ``(config_a, config_b)`` pairs
        plus the labels recorded on every output row so the correction
        is auditable (``n_comparisons_in_family`` = ``len(pairs)``).
    include_cohens_d:
        Adds the parametric Cohen's d column.  OFF by default: it has
        no sensible interpretation on 0/1-heavy LOO differences (see
        :func:`cohens_d_paired`).
    diff_ci:
        Adds the bootstrap CI of the PAIRED mean difference
        (``diff_mean``, ``diff_ci_lower``, ``diff_ci_upper``) — the CI
        that must agree with the Wilcoxon verdict (individual
        per-config CIs may overlap under a significant paired test).

    Returns
    -------
    pd.DataFrame
        One row per pair: ``family``, ``group``,
        ``n_comparisons_in_family``, ``config_a``, ``config_b``,
        ``mean_a``, ``mean_b``, ``statistic``, ``p_value``,
        ``corrected_p``, ``significant``, ``n_pairs``,
        ``n_nonzero_pairs``, plus the effect-size and paired-diff-CI
        columns enabled above.
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
    if pairs is None:
        configs = sorted(df["config"].unique())
        if len(configs) < 2:
            raise ValueError(
                f"Need at least 2 distinct configs for pairwise comparison, got {len(configs)}."
            )
        pairs = list(combinations(configs, 2))
    else:
        pairs = list(pairs)
        if not pairs:
            raise ValueError("pairs must be a non-empty list of (config_a, config_b) tuples.")

    needed = sorted({c for pair in pairs for c in pair})
    missing = [c for c in needed if c not in set(df["config"])]
    if missing:
        raise ValueError(f"configs referenced by pairs but absent from results: {missing}")

    pivot = df[df["config"].isin(needed)].pivot(index="user_id", columns="config", values=metric)

    rows: list[dict] = []
    raw_p_values: list[float] = []
    m_family = len(pairs)

    for config_a, config_b in pairs:
        valid = pivot[[config_a, config_b]].dropna()
        scores_a = valid[config_a].to_numpy()
        scores_b = valid[config_b].to_numpy()

        stat, p_val = wilcoxon_test(scores_a, scores_b)
        raw_p_values.append(p_val)

        row = {
            "family": family,
            "group": group,
            "n_comparisons_in_family": m_family,
            "config_a": config_a,
            "config_b": config_b,
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "statistic": stat,
            "p_value": p_val,
            "n_pairs": int(len(scores_a)),
            # How many pairs carry signal: with 0/1 per-user metrics most
            # differences are zero; report it so n_pairs is not read as
            # the effective sample size.
            "n_nonzero_pairs": n_nonzero_pairs(scores_a, scores_b),
        }
        if diff_ci:
            diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(
                scores_a,
                scores_b,
                n_iterations=n_iterations,
                alpha=alpha,
                seed=seed,
            )
            row["diff_mean"] = diff_mean
            row["diff_ci_lower"] = diff_lo
            row["diff_ci_upper"] = diff_hi
        if include_effect_size:
            delta = cliffs_delta(scores_a, scores_b)
            row["cliffs_delta"] = delta
            row["cliffs_magnitude"] = cliffs_delta_magnitude(delta)
            if include_cohens_d:
                row["cohens_d"] = cohens_d_paired(scores_a, scores_b)
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
