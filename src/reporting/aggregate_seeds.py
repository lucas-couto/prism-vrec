"""Cross-seed aggregation of long-format evaluation outputs.

A multi-seed run (``seeds: [42, 99, 7]`` in ``configs/default.yaml`` or
``--seeds 42,99,7`` at the CLI) executes the pipeline once per seed
under ``<paths.results>_seed{N}/``.  This module reads each seed's
long-format CSVs and writes a consolidated table with ``mean``,
``std``, ``median``, ``n_seeds`` per (dataset, recommender, extractor,
fusion, condition, metric, k) cell — the headline number researchers
actually report.

Pure pandas (no torch / no ML deps); safe to run on a laptop.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
        mean_across_seeds="mean",
        std_across_seeds=lambda s: s.std(ddof=1),
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

    return written
