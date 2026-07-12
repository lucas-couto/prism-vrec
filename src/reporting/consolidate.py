"""Read granular ``results/tables/`` CSVs and emit long-format files.

The transformation logic lives in :mod:`src.reporting.long_format`;
this module orchestrates filesystem IO (globbing + writing the three
consolidated CSVs).  Pipeline steps call ``write_consolidated`` after
their normal outputs so the long-format files are always in sync.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.long_format import (
    classify_table_file,
    evaluation_to_long,
    friedman_to_long,
    pairwise_to_long,
    summary_to_long,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Kept in sync with the built-in recommenders so a config is never
# mislabelled "unknown" if the registry import ever fails.  Any plugin
# recommender is picked up via the live registry below; this list is
# only the last-resort fallback.
_BUILTIN_RECOMMENDERS = ["acf", "avbpr", "bpr", "deepstyle", "vbpr", "vnpr"]


def _known_recommenders() -> list[str]:
    """Pull the recommender registry, falling back to the built-in list."""
    try:
        from src.recommenders.registry import registered_recommender_names

        return list(registered_recommender_names())
    except Exception:  # noqa: BLE001
        logger.warning("Recommender registry unavailable; using built-in fallback list.")
        return list(_BUILTIN_RECOMMENDERS)


def consolidate_evaluation(tables_dir: Path) -> pd.DataFrame:
    """Aggregate per-user evaluation CSVs into one row per cell × metric × k."""
    frames: list[pd.DataFrame] = []
    for path in sorted(tables_dir.glob("*_evaluation_*.csv")):
        info = classify_table_file(path)
        if info is None or info["kind"] != "evaluation":
            continue
        eval_df = pd.read_csv(path)
        long_df = evaluation_to_long(
            eval_df,
            dataset=info["dataset"],
            condition=info["condition"],
        )
        if long_df.empty:
            continue
        aggregated = _aggregate_per_user(long_df)
        frames.append(aggregated)
        logger.info("  evaluation: %s rows from %s", len(aggregated), path.name)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_per_user(long_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-user rows into one row per cell with ``mean`` + ``n_users``."""
    group_keys = [
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
    group_keys = [k for k in group_keys if k in long_df.columns]
    return (
        long_df.groupby(group_keys, dropna=False)
        .agg(n_users=("value", "size"), mean=("value", "mean"))
        .reset_index()
    )


def consolidate_bootstrap(
    tables_dir: Path,
    known_recommenders: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every ``_summary_*.csv`` into one long table."""
    recs = known_recommenders or _known_recommenders()
    frames: list[pd.DataFrame] = []
    for path in sorted(tables_dir.glob("*_summary_*.csv")):
        info = classify_table_file(path)
        if info is None or info["kind"] != "summary":
            continue
        long_df = summary_to_long(
            pd.read_csv(path),
            dataset=info["dataset"],
            metric=info["metric"],
            k=int(info["k"]),
            known_recommenders=recs,
        )
        if not long_df.empty:
            frames.append(long_df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info("  bootstrap_ci: %d rows from %d files", len(out), len(frames))
    return out


def consolidate_statistical_tests(
    tables_dir: Path,
    known_recommenders: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate Friedman + pairwise Wilcoxon CSVs into one long table."""
    recs = known_recommenders or _known_recommenders()
    frames: list[pd.DataFrame] = []

    for path in sorted(tables_dir.glob("*_friedman_*.csv")):
        info = classify_table_file(path)
        if info is None or info["kind"] != "friedman":
            continue
        long_df = friedman_to_long(
            pd.read_csv(path),
            dataset=info["dataset"],
            metric=info["metric"],
            k=int(info["k"]),
        )
        if not long_df.empty:
            frames.append(long_df)

    for path in sorted(tables_dir.glob("*_pairwise_*.csv")):
        info = classify_table_file(path)
        if info is None or info["kind"] != "pairwise":
            continue
        long_df = pairwise_to_long(
            pd.read_csv(path),
            dataset=info["dataset"],
            metric=info["metric"],
            k=int(info["k"]),
            known_recommenders=recs,
        )
        if not long_df.empty:
            frames.append(long_df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info("  statistical_tests: %d rows from %d files", len(out), len(frames))
    return out


def write_consolidated(
    tables_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Run the three consolidations and write the resulting CSVs.

    The three outputs are written to ``output_dir`` (defaults to
    ``tables_dir``).  Empty results yield an empty CSV — the caller can
    detect "nothing to consolidate" by checking row counts later.
    """
    out_dir = output_dir or tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    known_recs = _known_recommenders()

    logger.info("Consolidating evaluation...")
    eval_long = consolidate_evaluation(tables_dir)
    eval_path = out_dir / "evaluation_aggregated.csv"
    eval_long.to_csv(eval_path, index=False)

    logger.info("Consolidating bootstrap CIs...")
    ci_long = consolidate_bootstrap(tables_dir, known_recs)
    ci_path = out_dir / "bootstrap_ci.csv"
    ci_long.to_csv(ci_path, index=False)

    logger.info("Consolidating statistical tests...")
    tests_long = consolidate_statistical_tests(tables_dir, known_recs)
    tests_path = out_dir / "statistical_tests.csv"
    tests_long.to_csv(tests_path, index=False)

    return {
        "evaluation_aggregated": eval_path,
        "bootstrap_ci": ci_path,
        "statistical_tests": tests_path,
    }
