"""Cross-seed aggregation of long-format evaluation outputs.

A multi-seed run (``seeds: [42, 99, 7]`` in ``configs/default.yaml`` or
``--seeds 42,99,7`` at the CLI) executes the pipeline once per seed
under ``<paths.results>_seed{N}/``.  This module reads each seed's
long-format CSVs and writes a consolidated table with ``mean``,
``std``, ``median``, ``n_seeds`` per (dataset, recommender, extractor,
fusion, condition, metric, k) cell — the headline number researchers
actually report.

Cross-seed reconciliation of the statistical tests
--------------------------------------------------

Every p-value in a per-seed ``statistical_tests.csv`` is a statement
about ONE training realisation: it tests whether config A beats config
B *for the models trained under that seed*, not whether method A beats
method B in general.  ``aggregate_statistical_tests`` therefore builds
``statistical_tests_across_seeds.csv`` — per (dataset, family, group,
pair, metric, k): ``n_seeds``, ``n_seeds_significant`` (Holm-corrected
verdicts), the median paired difference and its sign agreement across
seeds, and min/max/median of the Holm-corrected p-value.  This is a
DESCRIPTIVE reconciliation, deliberately not a p-value combination
method (no Fisher's method): a method-level claim requires the
Holm-corrected verdict AND the sign of the difference to agree across
seeds, and must be phrased as such.

Pure pandas (no torch / no ML deps); safe to run on a laptop.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_GROUP_KEYS_EVAL = [
    "dataset",
    "file_condition",
    "recommender",
    "embedding_name",
    "extractor",
    "fusion",
    "condition",
    "embedding_dim",
    "metric",
    "k",
]

_GROUP_KEYS_CI = [
    "dataset",
    "recommender",
    "extractor",
    "fusion",
    "condition",
    "embedding_dim",
    "metric",
    "k",
]

# Pair identity in the long-format statistical_tests.csv.  ``config_a``
# / ``config_b`` are THE identity — the full config strings, including
# the ``_D<dim>`` token.  The parsed components that follow are kept as
# descriptive columns but do NOT suffice as a key: configs differing
# only in embedding_dim parse to the same components and would collapse
# into one group, making ``n_seeds`` count seeds × dims (R2).
_GROUP_KEYS_TESTS = [
    "dataset",
    "family",
    "group",
    "config_a",
    "config_b",
    "recommender_a",
    "extractor_a",
    "fusion_a",
    "condition_a",
    "recommender_b",
    "extractor_b",
    "fusion_b",
    "condition_b",
    "metric",
    "k",
]


def _read_per_seed(
    seed_dirs: list[Path],
    filename: str,
    seeds: list[int] | None,
) -> pd.DataFrame:
    """Concatenate ``<seed_dir>/tables/<filename>`` from each seed dir.

    A missing file in a particular seed dir is skipped with a warning;
    callers should check ``len(df)`` before aggregating.  Each frame is
    tagged with a ``seed`` column so the caller can drop duplicates or
    audit which seeds contributed to a given cell.
    """
    frames: list[pd.DataFrame] = []
    seeds_tag = seeds or [None] * len(seed_dirs)
    for seed_dir, seed in zip(seed_dirs, seeds_tag, strict=False):
        path = Path(seed_dir) / "tables" / filename
        if not path.exists():
            logger.warning("Per-seed CSV missing: %s", path)
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if seed is not None:
            df["seed"] = seed
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate(df: pd.DataFrame, group_keys: list[str], value_col: str) -> pd.DataFrame:
    """Group ``df`` by ``group_keys`` and reduce ``value_col`` to descriptive stats.

    Returns columns ``mean_across_seeds``, ``std_across_seeds``,
    ``median_across_seeds``, ``min_across_seeds``,
    ``max_across_seeds``, ``n_seeds``.  ``std`` uses the sample
    convention (ddof=1) and is ``NaN`` when only one seed contributed.
    """
    if df.empty:
        return df
    keys = [k for k in group_keys if k in df.columns]
    grouped = df.groupby(keys, dropna=False)[value_col]
    return grouped.agg(
        # "std" already uses ddof=1 (NaN for a single seed) and takes the
        # fast cython path, unlike a per-group python lambda.
        mean_across_seeds="mean",
        std_across_seeds="std",
        median_across_seeds="median",
        min_across_seeds="min",
        max_across_seeds="max",
        n_seeds="count",
    ).reset_index()


def aggregate_evaluation(seed_dirs: list[Path], seeds: list[int] | None) -> pd.DataFrame:
    """Aggregate per-seed ``evaluation_aggregated.csv`` files."""
    df = _read_per_seed(seed_dirs, "evaluation_aggregated.csv", seeds)
    if df.empty:
        return df
    return _aggregate(df, _GROUP_KEYS_EVAL, "mean")


def aggregate_bootstrap_ci(seed_dirs: list[Path], seeds: list[int] | None) -> pd.DataFrame:
    """Aggregate per-seed ``bootstrap_ci.csv`` files.

    The CI bounds themselves are not re-bootstrapped across seeds;
    instead we report the across-seed dispersion of the per-seed
    bootstrap means.  Researchers wanting a true multi-level CI should
    re-run bootstrap on the pooled per-user scores — out of scope here.
    """
    df = _read_per_seed(seed_dirs, "bootstrap_ci.csv", seeds)
    if df.empty:
        return df
    return _aggregate(df, _GROUP_KEYS_CI, "mean")


def _sign_agreement(diffs: pd.Series) -> float:
    """Fraction of seeds whose difference sign matches the median's sign.

    ``1.0`` means every seed agrees on the direction of the effect;
    ``NaN`` when no seed reported a difference.
    """
    values = diffs.dropna()
    if values.empty:
        return float("nan")
    return float((np.sign(values) == np.sign(values.median())).mean())


def aggregate_statistical_tests(seed_dirs: list[Path], seeds: list[int] | None) -> pd.DataFrame:
    """Reconcile per-seed pairwise verdicts (see the module docstring).

    Reads each seed's ``statistical_tests.csv``, keeps the Wilcoxon
    rows, and returns one row per (dataset, family, group, pair,
    metric, k) with ``n_seeds``, ``n_seeds_significant``
    (Holm-corrected verdicts), ``median_diff_mean``,
    ``sign_agreement``, and ``p_holm_min/median/max``.  Descriptive
    only — no p-value combination is performed.
    """
    df = _read_per_seed(seed_dirs, "statistical_tests.csv", seeds)
    if df.empty:
        return df
    if "test_type" in df.columns:
        df = df[df["test_type"] == "wilcoxon"].copy()
    if df.empty:
        return pd.DataFrame()
    if "diff_mean" not in df.columns:
        df["diff_mean"] = np.nan
    if "config_a" not in df.columns or "config_b" not in df.columns:
        # Legacy per-seed CSVs (pre config_a/config_b columns): grouping
        # falls back to the parsed components, which collapse configs
        # differing only in embedding_dim (R2).  Warn loudly instead of
        # misgrouping silently.
        logger.warning(
            "statistical_tests.csv lacks config_a/config_b; falling back "
            "to parsed-component grouping. Configs differing only in "
            "embedding_dim WILL collapse into one group — regenerate the "
            "per-seed long-format tables to fix the pair identity."
        )

    keys = [k for k in _GROUP_KEYS_TESTS if k in df.columns]
    grouped = df.groupby(keys, dropna=False)
    return grouped.agg(
        n_seeds=("significant", "size"),
        n_seeds_significant=("significant", "sum"),
        median_diff_mean=("diff_mean", "median"),
        sign_agreement=("diff_mean", _sign_agreement),
        p_holm_min=("corrected_p", "min"),
        p_holm_median=("corrected_p", "median"),
        p_holm_max=("corrected_p", "max"),
    ).reset_index()


def write_cross_seed_aggregates(
    seed_dirs: list[Path],
    output_dir: Path,
    seeds: list[int] | None = None,
) -> dict[str, Path]:
    """Read per-seed long-format CSVs and write the consolidated aggregates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    eval_df = aggregate_evaluation(seed_dirs, seeds)
    eval_path = output_dir / "evaluation_multi_seed.csv"
    eval_df.to_csv(eval_path, index=False)
    written["evaluation_multi_seed"] = eval_path
    logger.info(
        "Wrote cross-seed evaluation (%d rows) to %s",
        len(eval_df),
        eval_path,
    )

    ci_df = aggregate_bootstrap_ci(seed_dirs, seeds)
    ci_path = output_dir / "bootstrap_ci_multi_seed.csv"
    ci_df.to_csv(ci_path, index=False)
    written["bootstrap_ci_multi_seed"] = ci_path
    logger.info(
        "Wrote cross-seed bootstrap CI (%d rows) to %s",
        len(ci_df),
        ci_path,
    )

    tests_df = aggregate_statistical_tests(seed_dirs, seeds)
    tests_path = output_dir / "statistical_tests_across_seeds.csv"
    tests_df.to_csv(tests_path, index=False)
    written["statistical_tests_across_seeds"] = tests_path
    logger.info(
        "Wrote cross-seed statistical reconciliation (%d rows) to %s",
        len(tests_df),
        tests_path,
    )

    return written
