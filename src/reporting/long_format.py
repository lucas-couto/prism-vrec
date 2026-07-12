"""Long-format consolidation for evaluation and statistical outputs.

The pipeline writes one CSV per metric × test-type × dataset (~160 files
total).  This module reshapes those into three normalised long-format
tables suitable for thesis writing, plotting, and downstream analysis::

    evaluation_long.csv      per-user metric scores
    bootstrap_ci.csv         per-config bootstrap means + confidence intervals
    statistical_tests.csv    Friedman + pairwise Wilcoxon test rows

Each table follows tidy-data conventions (one row per observation,
identifier columns instead of identifier filenames) so a single
``pd.read_csv`` + ``DataFrame.query`` replaces opening dozens of files.

The original per-metric CSVs are kept intact — this module reads them
and produces new files alongside; it never overwrites.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Filename patterns the pipeline emits in ``results/tables/``.
_EVAL_PATTERN = re.compile(r"^(?P<ds>.+?)_evaluation_(?P<cond>frozen|finetuned)\.csv$")
# [a-z0-9]+ (not [a-z]+) so metrics with a digit like ``f1`` classify;
# otherwise an f1 table would glob-match but silently fail to parse.
_SUMMARY_PATTERN = re.compile(r"^(?P<ds>.+?)_summary_(?P<metric>[a-z0-9]+)_at_(?P<k>\d+)\.csv$")
_FRIEDMAN_PATTERN = re.compile(r"^(?P<ds>.+?)_friedman_(?P<metric>[a-z0-9]+)_at_(?P<k>\d+)\.csv$")
_PAIRWISE_PATTERN = re.compile(r"^(?P<ds>.+?)_pairwise_(?P<metric>[a-z0-9]+)_at_(?P<k>\d+)\.csv$")

_METRIC_COL_PATTERN = re.compile(r"^(?P<metric>precision|recall|ndcg|map|f1)@(?P<k>\d+)$")
_EMBEDDING_DIM_PATTERN = re.compile(r"_D(\d+)$")
_FINETUNED_SUFFIX = "_finetuned"
_HYBRID_PREFIX = "hybrid_"


def parse_embedding_name(name: str) -> dict[str, object]:
    """Split an ``embedding_name`` into its semantic components.

    Examples
    --------
    ``clip_vitb32_D128``          -> extractor=clip_vitb32, fusion=none, condition=frozen
    ``vit_b16_finetuned_D128``    -> extractor=vit_b16, fusion=none, condition=finetuned
    ``hybrid_adaptive_gated_D128`` -> extractor=hybrid, fusion=adaptive_gated, condition=frozen
    ``hybrid_mean_finetuned_D128`` -> extractor=hybrid, fusion=mean, condition=finetuned
    ``none``                      -> extractor=none, fusion=none, condition=both
    """
    if name == "none":
        return {"extractor": "none", "fusion": "none", "condition": "both", "embedding_dim": None}

    stem = name
    embedding_dim: int | None = None
    dim_match = _EMBEDDING_DIM_PATTERN.search(stem)
    if dim_match is not None:
        embedding_dim = int(dim_match.group(1))
        stem = stem[: dim_match.start()]

    condition = "frozen"
    if stem.endswith(_FINETUNED_SUFFIX):
        condition = "finetuned"
        stem = stem[: -len(_FINETUNED_SUFFIX)]

    if stem.startswith(_HYBRID_PREFIX):
        extractor = "hybrid"
        fusion = stem[len(_HYBRID_PREFIX) :]
    else:
        extractor = stem
        fusion = "none"

    return {
        "extractor": extractor,
        "fusion": fusion,
        "condition": condition,
        "embedding_dim": embedding_dim,
    }


def parse_config(config: str, known_recommenders: Iterable[str]) -> dict[str, object]:
    """Split a ``config`` string into recommender + embedding components.

    ``config`` is ``{recommender}_{embedding_name}`` as produced by
    ``_ensure_config`` in :mod:`src.evaluation.statistical`.  The
    recommender prefix may itself contain underscores (e.g.
    ``uniform_noise``) so we match against the registry, longest first,
    to find the correct boundary.
    """
    candidates = sorted(set(known_recommenders), key=len, reverse=True)
    for candidate in candidates:
        if config == candidate:
            return {"recommender": candidate, **parse_embedding_name("none")}
        if config.startswith(candidate + "_"):
            embedding = config[len(candidate) + 1 :]
            return {"recommender": candidate, **parse_embedding_name(embedding)}

    return {"recommender": "unknown", **parse_embedding_name(config)}


def evaluation_to_long(
    eval_df: pd.DataFrame,
    *,
    dataset: str,
    condition: str,
) -> pd.DataFrame:
    """Melt one ``{ds}_evaluation_{condition}.csv`` into long format.

    Input has one row per (user, model, embedding) with wide metric
    columns (``precision@5``, ``ndcg@10`` etc).  Output has one row per
    (user, model, embedding, metric, k).
    """
    metric_cols = [c for c in eval_df.columns if _METRIC_COL_PATTERN.match(c)]
    if not metric_cols:
        logger.warning("No metric columns found for %s/%s", dataset, condition)
        return pd.DataFrame()

    id_cols = [c for c in eval_df.columns if c not in metric_cols]
    melted = eval_df.melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="metric_at_k",
        value_name="value",
    )
    metric_parsed = melted["metric_at_k"].str.extract(_METRIC_COL_PATTERN)
    melted["metric"] = metric_parsed["metric"]
    melted["k"] = metric_parsed["k"].astype(int)
    melted = melted.drop(columns=["metric_at_k"])

    embedding_parts = melted["embedding_name"].apply(parse_embedding_name).apply(pd.Series)
    melted = pd.concat([melted, embedding_parts], axis=1)

    melted["dataset"] = dataset
    melted["file_condition"] = condition
    melted = melted.rename(columns={"model_name": "recommender"})

    keep = [
        "dataset",
        "file_condition",
        "recommender",
        "embedding_name",
        "extractor",
        "fusion",
        "condition",
        "embedding_dim",
        "user_id",
        "metric",
        "k",
        "value",
        # v2 provenance: evaluation protocol, visual input dim consumed by
        # the model's E, and trainable-parameter count.
        "protocol",
        "visual_input_dim",
        "n_trainable_params",
    ]
    return melted.loc[:, [c for c in keep if c in melted.columns]]


def summary_to_long(
    summary_df: pd.DataFrame,
    *,
    dataset: str,
    metric: str,
    k: int,
    known_recommenders: Iterable[str],
) -> pd.DataFrame:
    """Reshape a per-config bootstrap summary into long format.

    Input has one row per ``config`` (= ``{recommender}_{embedding_name}``);
    output adds ``recommender``, ``extractor``, ``fusion``, ``condition``
    columns plus the dataset/metric/k identifiers.
    """
    if summary_df.empty:
        return summary_df

    parsed = (
        summary_df["config"].apply(lambda c: parse_config(c, known_recommenders)).apply(pd.Series)
    )
    out = pd.concat([summary_df, parsed], axis=1)
    out["dataset"] = dataset
    out["metric"] = metric
    out["k"] = int(k)

    keep = [
        "dataset",
        "recommender",
        "extractor",
        "fusion",
        "condition",
        "embedding_dim",
        "metric",
        "k",
        "n_users",
        "mean",
        "ci_lower",
        "ci_upper",
        "ci_width",
    ]
    return out.loc[:, [c for c in keep if c in out.columns]]


def friedman_to_long(
    friedman_df: pd.DataFrame,
    *,
    dataset: str,
    metric: str,
    k: int,
) -> pd.DataFrame:
    """Tag a single-row Friedman test with dataset/metric/k identifiers."""
    if friedman_df.empty:
        return friedman_df

    out = friedman_df.copy()
    out["dataset"] = dataset
    out["metric"] = metric
    out["k"] = int(k)
    out["test_type"] = "friedman"

    keep = [
        "dataset",
        "metric",
        "k",
        "test_type",
        "statistic",
        "p_value",
        "significant",
        "n_configs",
        "n_users",
        "note",
    ]
    return out.loc[:, [c for c in keep if c in out.columns]]


def pairwise_to_long(
    pairwise_df: pd.DataFrame,
    *,
    dataset: str,
    metric: str,
    k: int,
    known_recommenders: Iterable[str],
) -> pd.DataFrame:
    """Reshape a pairwise Wilcoxon table with parsed config_a/config_b."""
    if pairwise_df.empty:
        return pairwise_df

    parsed_a = (
        pairwise_df["config_a"]
        .apply(lambda c: parse_config(c, known_recommenders))
        .apply(pd.Series)
        .add_suffix("_a")
    )
    parsed_b = (
        pairwise_df["config_b"]
        .apply(lambda c: parse_config(c, known_recommenders))
        .apply(pd.Series)
        .add_suffix("_b")
    )
    out = pd.concat([pairwise_df, parsed_a, parsed_b], axis=1)
    out["dataset"] = dataset
    out["metric"] = metric
    out["k"] = int(k)
    out["test_type"] = "wilcoxon"

    keep = [
        "dataset",
        "metric",
        "k",
        "test_type",
        "recommender_a",
        "extractor_a",
        "fusion_a",
        "condition_a",
        "recommender_b",
        "extractor_b",
        "fusion_b",
        "condition_b",
        "statistic",
        "p_value",
        "corrected_p",
        "significant",
        "mean_a",
        "mean_b",
        "cohens_d",
        "cliffs_delta",
        "cliffs_magnitude",
    ]
    return out.loc[:, [c for c in keep if c in out.columns]]


def classify_table_file(path: Path) -> dict[str, str] | None:
    """Return ``{kind, dataset, metric, k}`` for a tables-dir CSV, or None."""
    name = path.name
    if m := _EVAL_PATTERN.match(name):
        return {"kind": "evaluation", "dataset": m["ds"], "condition": m["cond"]}
    if m := _SUMMARY_PATTERN.match(name):
        return {"kind": "summary", "dataset": m["ds"], "metric": m["metric"], "k": m["k"]}
    if m := _FRIEDMAN_PATTERN.match(name):
        return {"kind": "friedman", "dataset": m["ds"], "metric": m["metric"], "k": m["k"]}
    if m := _PAIRWISE_PATTERN.match(name):
        return {"kind": "pairwise", "dataset": m["ds"], "metric": m["metric"], "k": m["k"]}
    return None
