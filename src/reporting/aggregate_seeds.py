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
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.paired_validation import ObservationConflictError

logger = logging.getLogger(__name__)

_SEED_DIR_PATTERN = re.compile(r"_seed(\d+)$")

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

# ``report_condition`` is the partition a statistical file came from
# (``frozen`` / ``all`` / ``frozen_restricted`` ...); the same pair tested
# in two partitions is two comparisons, never twice the seeds (R05).
_GROUP_KEYS_CI = [
    "dataset",
    "report_condition",
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
    "report_condition",
    "population_policy",
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
    tagged with a ``seed`` column (given, or parsed from a
    ``..._seed<N>`` directory name) so the aggregation counts DISTINCT
    seeds: the same seed read twice (a duplicated source file) is one
    observation, and two files that disagree about one seed are a
    conflict, never two seeds (R05).
    """
    frames: list[pd.DataFrame] = []
    seeds_tag = seeds or [_seed_from_dir(d) for d in seed_dirs]
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
    return _distinct_seed_rows(pd.concat(frames, ignore_index=True), filename)


def _seed_from_dir(seed_dir: Path) -> int | None:
    match = _SEED_DIR_PATTERN.search(Path(seed_dir).name)
    return int(match.group(1)) if match else None


def _distinct_seed_rows(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """One row per (identity, seed): drop identical repeats, fail on conflicts."""
    if "seed" not in df.columns:
        logger.warning(
            "%s: no seed identity (seeds not given and directory names carry no "
            "_seed<N> suffix); n_seeds will count rows, not distinct seeds.",
            filename,
        )
        return df
    value_columns = [c for c in df.columns if c in _VALUE_COLUMNS]
    key_columns = [c for c in df.columns if c not in value_columns]
    identical = df.duplicated(keep="first")
    conflicting = df[~identical].duplicated(subset=key_columns, keep=False)
    if conflicting.any():
        rows = df[~identical][conflicting]
        seeds = sorted(int(s) for s in rows["seed"].unique())
        raise ObservationConflictError(
            f"{filename}: {int(conflicting.sum())} row(s) disagree about the same cell under "
            f"seed {', '.join(str(s) for s in seeds)}; two sources claim one seed with "
            "different values. Refusing to count them as separate seeds."
        )
    if identical.any():
        logger.warning(
            "%s: %d identical duplicate row(s) across the seed sources ignored.",
            filename,
            int(identical.sum()),
        )
    return df[~identical].reset_index(drop=True)


#: Columns that carry measured values; every other column is identity.
_VALUE_COLUMNS = frozenset(
    {
        "mean",
        "n_users",
        "ci_lower",
        "ci_upper",
        "ci_width",
        "statistic",
        "p_value",
        "corrected_p",
        "significant",
        "n_pairs",
        "n_nonzero_pairs",
        "n_excluded_a",
        "n_excluded_b",
        "mean_a",
        "mean_b",
        "diff_mean",
        "diff_ci_lower",
        "diff_ci_upper",
        "cohens_d",
        "cliffs_delta",
        "n_wins",
        "n_losses",
        "n_ties",
        "pct_wins",
        "pct_losses",
        "pct_ties",
        "omnibus_significant",
        "n_configs",
        "note",
    }
)


def _aggregate(df: pd.DataFrame, group_keys: list[str], value_col: str) -> pd.DataFrame:
    """Group ``df`` by ``group_keys`` and reduce ``value_col`` to descriptive stats.

    Returns columns ``mean_across_seeds``, ``std_across_seeds``,
    ``median_across_seeds``, ``min_across_seeds``,
    ``max_across_seeds``, ``n_seeds``.  ``std`` uses the sample
    convention (ddof=1) and is ``NaN`` when only one seed contributed.
    ``n_seeds`` counts distinct seeds (see :func:`_read_per_seed`).
    """
    if df.empty:
        return df
    keys = [k for k in group_keys if k in df.columns]
    grouped = df.groupby(keys, dropna=False)
    stats = grouped[value_col].agg(
        # "std" already uses ddof=1 (NaN for a single seed) and takes the
        # fast cython path, unlike a per-group python lambda.
        mean_across_seeds="mean",
        std_across_seeds="std",
        median_across_seeds="median",
        min_across_seeds="min",
        max_across_seeds="max",
    )
    stats["n_seeds"] = _n_distinct_seeds(grouped, value_col)
    return stats.reset_index()


def _n_distinct_seeds(grouped, value_col: str) -> pd.Series:
    """Distinct ``seed`` values per group, or the row count without seed identity."""
    if "seed" in grouped.obj.columns:
        return grouped["seed"].nunique()
    return grouped[value_col].count()


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
    out = grouped.agg(
        n_seeds_significant=("significant", "sum"),
        median_diff_mean=("diff_mean", "median"),
        sign_agreement=("diff_mean", _sign_agreement),
        p_holm_min=("corrected_p", "min"),
        p_holm_median=("corrected_p", "median"),
        p_holm_max=("corrected_p", "max"),
    )
    out.insert(0, "n_seeds", _n_distinct_seeds(grouped, "significant"))
    return out.reset_index()


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
