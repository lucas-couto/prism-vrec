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
    _METRIC_COL_PATTERN,
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


#: Rows read per chunk when a per-user evaluation table is consolidated.
#: Melting the whole table first is what made this step unusable: a
#: 2.57 M-row table with 30 metric columns becomes a 77 M-row
#: intermediate on the way to a 2 025-row output, and the step was
#: OOM-killed at the container's 16 GB limit (found 2026-09-09).  The
#: melt is row-wise and the aggregation is a sum plus a count, so
#: chunking gives the same numbers at bounded memory.
EVALUATION_CHUNK_ROWS = 200_000

#: Cell identity in the consolidated evaluation file, in column order.
_GROUP_KEYS = (
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
)


#: What identifies a cell in the WIDE evaluation table.  Grouping by
#: "every column that is not a metric" instead exploded the group count
#: from 75 to 4 400: ``efd_excluded_frac@k`` and ``icov@k`` do not match
#: the metric pattern, and the first varies per user, so each user's
#: value became part of the key.  Those columns are dropped here exactly
#: as ``evaluation_to_long``'s fixed column list already dropped them.
_CELL_IDENTITY = ("dataset", "model_name", "embedding_name")


def _wide_totals(chunk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Per-cell ``(sums, non-null counts, row counts)`` of one WIDE chunk.

    Aggregating before the melt is the point.  Melting first turned a
    200 000-row chunk into 6 000 000 long rows and grouped them on ten
    keys, seven of which are text: chunking alone kept that inside the
    memory limit but not inside any usable time -- ten minutes on one
    table and still going (found 2026-09-09).  Collapsing the wide chunk
    to one row per cell first leaves ~75 rows to melt instead.

    ``sum`` / ``count`` skip missing values while ``size`` does not,
    which reproduces the original semantics exactly: ``n_users`` counts
    every user of the cell, and a mean ignores the users whose metric is
    undefined (EFD is ``nan`` when no recommended item has positive
    train popularity).
    """
    metric_cols = [c for c in chunk.columns if _METRIC_COL_PATTERN.match(c)]
    id_cols = [c for c in _CELL_IDENTITY if c in chunk.columns]
    grouped = chunk.groupby(id_cols, dropna=False)
    return (
        grouped[metric_cols].sum(min_count=1),
        grouped[metric_cols].count(),
        grouped.size(),
    )


def consolidate_evaluation(
    tables_dir: Path,
    *,
    datasets: set[str] | None = None,
    conditions: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-user evaluation CSVs into one row per cell x metric x k.

    ``datasets`` / ``conditions`` restrict the sweep to the run's own
    scope.  Without them the step globbed the whole tables directory and
    mixed a superseded run into the consolidated file: on 2026-09-09 a
    ``frozen`` amazon_men run also consolidated amazon_fashion
    ``finetuned`` tables written two days earlier.  ``None`` keeps the
    historical sweep-everything behaviour for callers with no scope.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(tables_dir.glob("*_evaluation_*.csv")):
        info = classify_table_file(path)
        if info is None or info["kind"] != "evaluation":
            continue
        if datasets is not None and info["dataset"] not in datasets:
            logger.info("  evaluation: %s outside this run's datasets - skipped.", path.name)
            continue
        if conditions is not None and info["condition"] not in conditions:
            logger.info("  evaluation: %s outside this run's conditions - skipped.", path.name)
            continue
        sums = counts = sizes = None
        for chunk in pd.read_csv(path, chunksize=EVALUATION_CHUNK_ROWS):
            chunk_sums, chunk_counts, chunk_sizes = _wide_totals(chunk)
            sums = chunk_sums if sums is None else sums.add(chunk_sums, fill_value=0)
            counts = chunk_counts if counts is None else counts.add(chunk_counts, fill_value=0)
            sizes = chunk_sizes if sizes is None else sizes.add(chunk_sizes, fill_value=0)
        if sums is None or sums.empty:
            continue
        means = (sums / counts.where(counts > 0)).reset_index()
        aggregated = evaluation_to_long(
            means,
            dataset=info["dataset"],
            condition=info["condition"],
        ).rename(columns={"value": "mean"})
        if aggregated.empty:
            continue
        # ``evaluation_to_long`` keeps a fixed column list, so the user
        # count rides back in by cell identity rather than through it.
        # ``Series.add`` across chunks promotes the count to float.
        per_cell = (
            sizes.astype("int64")
            .rename("n_users")
            .reset_index()
            .rename(columns={"model_name": "recommender"})
        )
        on = [c for c in per_cell.columns if c != "n_users" and c in aggregated.columns]
        aggregated = aggregated.merge(per_cell[[*on, "n_users"]], on=on, how="left")
        keys = [k for k in (*_GROUP_KEYS, "n_users") if k in aggregated.columns]
        aggregated = aggregated[[*keys, "mean"]]
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


def _iter_metric_groups(df: pd.DataFrame):
    """Yield ``((metric, k), sub_frame)`` from a consolidated per-dataset CSV.

    The per-dataset files carry ``metric`` + ``k`` identity columns (one
    file per kind since 2.9.0); the ``*_to_long`` converters keep their
    per-metric contract, so this splits the file back into the shape
    they expect, with the identity columns dropped from the sub-frame.
    """
    if df.empty or "metric" not in df.columns:
        return
    for (metric, k), sub in df.groupby(["metric", "k"], dropna=False):
        yield (str(metric), int(k)), sub.drop(columns=["metric", "k"]).reset_index(drop=True)


def consolidate_bootstrap(
    tables_dir: Path,
    known_recommenders: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every ``_summary_*.csv`` into one long table."""
    recs = known_recommenders or _known_recommenders()
    frames: list[pd.DataFrame] = []
    for path in sorted(tables_dir.glob("*_summary.csv")):
        info = classify_table_file(path)
        if info is None or info["kind"] != "summary":
            continue
        for (metric, k), sub in _iter_metric_groups(pd.read_csv(path)):
            long_df = summary_to_long(
                sub,
                dataset=info["dataset"],
                metric=metric,
                k=k,
                known_recommenders=recs,
                report_condition=info["report_condition"],
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

    for path in sorted(tables_dir.glob("*_friedman.csv")):
        info = classify_table_file(path)
        if info is None or info["kind"] != "friedman":
            continue
        for (metric, k), sub in _iter_metric_groups(pd.read_csv(path)):
            long_df = friedman_to_long(
                sub,
                dataset=info["dataset"],
                metric=metric,
                k=k,
                report_condition=info["report_condition"],
            )
            if not long_df.empty:
                frames.append(long_df)

    for path in sorted(tables_dir.glob("*_pairwise.csv")):
        info = classify_table_file(path)
        if info is None or info["kind"] != "pairwise":
            continue
        for (metric, k), sub in _iter_metric_groups(pd.read_csv(path)):
            long_df = pairwise_to_long(
                sub,
                dataset=info["dataset"],
                metric=metric,
                k=k,
                known_recommenders=recs,
                report_condition=info["report_condition"],
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
    *,
    datasets: set[str] | None = None,
    conditions: set[str] | None = None,
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
    eval_long = consolidate_evaluation(tables_dir, datasets=datasets, conditions=conditions)
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
