"""Step 07 — Statistical reporting.

For every dataset and every PRIMARY metric this step produces three
artefacts:

* ``{dataset}_summary_{metric}.csv``
  Per-config mean with bootstrap confidence intervals (descriptive;
  the inference below is PAIRED — overlapping individual CIs do not
  contradict a significant paired test).

* ``{dataset}_friedman_{metric}.csv``
  Friedman omnibus test PER COMPARISON FAMILY — answers "is anyone
  different within this family?" before its pairwise tests.

* ``{dataset}_pairwise_{metric}.csv``
  Wilcoxon signed-rank tests with the multiple-comparison correction
  (Holm by default) applied WITHIN each comparison family — the set of
  hypotheses one research question defines (see
  :mod:`src.evaluation.comparison_families`), never the Cartesian
  product of every config.  Every row carries ``family``, ``group``
  and ``n_comparisons_in_family`` so the correction is auditable.
  Effect size: Cliff's delta (primary); the paired-difference
  bootstrap CI accompanies each pair.

**Primary metrics.** Under the leave-one-out protocol there are only
two independent signals per user: *hit or not* (recall@k ≡ HitRate@k)
and *at which rank* (ndcg@k; map@k with one relevant item ≡ MRR-style
``1/rank``; precision@k = recall@k / k; F1 is derived from both).  The
step therefore reports ``recall`` and ``ndcg`` by default; the derived
metrics remain in the raw evaluation CSVs and can be included with
``include_derived_metrics: true`` — they must not be read as
independent evidence.

When per-user metrics are not available (the evaluator was run with
aggregated outputs only), the step falls back to writing a comparative
table without inferential statistics and logs a warning.

Configuration knobs (``configs/evaluation.yaml`` -> ``statistical:``):

* ``alpha``                       — family-wise significance level (default 0.05)
* ``correction``                  — ``"holm"`` (default), ``"bonferroni"``, ``"none"``
* ``families``                    — comparison families to compute (default: the four
                                    question-aligned families; add ``"all_pairs"`` for
                                    the exploratory full product)
* ``primary_metrics``             — metric families reported (default ``[recall, ndcg]``)
* ``include_derived_metrics``     — also analyse precision/map (default false)
* ``bootstrap.enabled``           — toggle bootstrap CIs (default true)
* ``bootstrap.n_iterations``      — number of resamples (default 1000)
* ``friedman.enabled``            — toggle Friedman test (default true)
* ``effect_size``                 — toggle Cliff's delta columns (default true)
* ``include_cohens_d``            — add parametric Cohen's d (default false; see
                                    ``cohens_d_paired`` docstring for why)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.comparison_families import (
    DEFAULT_FAMILIES,
    enumerate_family_instances,
)
from src.evaluation.statistical import (
    friedman_test,
    pairwise_significance,
    per_model_summary,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run(condition: str = "frozen") -> None:
    """Run the statistical analyses for the given condition.

    ``condition`` may be ``frozen``, ``finetuned``, or ``all``.  The
    ``all`` mode merges frozen+finetuned evaluation tables before
    running the analyses so the resulting CSVs compare both batteries
    side-by-side.
    """
    if condition not in {"frozen", "finetuned", "all"}:
        raise ValueError(f"condition must be 'frozen', 'finetuned' or 'all', got {condition!r}")

    config = load_config()
    datasets = config.get("datasets", [])
    if not datasets:
        logger.info("statistical step skipped: datasets list is empty in configs/default.yaml.")
        return
    k_values = config.get("k_values", [5, 10, 20])

    stat_cfg = config.get("statistical", {})
    alpha = stat_cfg.get("alpha", 0.05)
    correction = stat_cfg.get("correction", "holm")
    families = stat_cfg.get("families", list(DEFAULT_FAMILIES))
    primary_metrics = stat_cfg.get("primary_metrics", ["recall", "ndcg"])
    include_derived = stat_cfg.get("include_derived_metrics", False)
    bootstrap_cfg = stat_cfg.get("bootstrap", {})
    bootstrap_enabled = bootstrap_cfg.get("enabled", True)
    bootstrap_iters = bootstrap_cfg.get("n_iterations", 1000)
    friedman_enabled = stat_cfg.get("friedman", {}).get("enabled", True)
    effect_size = stat_cfg.get("effect_size", True)
    include_cohens_d = stat_cfg.get("include_cohens_d", False)

    results_dir = Path(config.get("paths", {}).get("results", "results")) / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Condition: %s  alpha=%.3f  correction=%s  families=%s  "
        "primary_metrics=%s  bootstrap=%s  friedman=%s  effect_size=%s",
        condition,
        alpha,
        correction,
        families,
        primary_metrics,
        bootstrap_enabled,
        friedman_enabled,
        effect_size,
    )

    for dataset_name in datasets:
        logger.info("=== Dataset: %s ===", dataset_name)
        eval_df = _load_evaluation(results_dir, dataset_name, condition)
        if eval_df is None:
            continue

        metrics = _metrics_to_test(eval_df, k_values, primary_metrics, include_derived)
        if not metrics:
            logger.warning("  No supported metric columns found in evaluation file.")
            continue

        per_user = "user_id" in eval_df.columns
        instances = enumerate_family_instances(eval_df, families) if per_user else []
        if per_user and not instances:
            logger.warning(
                "  No comparison-family instance matched the available configs; "
                "nothing to test (families=%s).",
                families,
            )

        for metric in metrics:
            logger.info("  --- %s ---", metric)

            if per_user:
                if bootstrap_enabled:
                    summary = per_model_summary(
                        eval_df,
                        metric=metric,
                        n_iterations=bootstrap_iters,
                        alpha=alpha,
                    )
                    out = results_dir / f"{dataset_name}_summary_{_safe(metric)}.csv"
                    summary.to_csv(out, index=False)
                    logger.info("    bootstrap CI saved: %s", out)

                if friedman_enabled and instances:
                    fried_rows = []
                    for inst in instances:
                        fried = friedman_test(
                            eval_df,
                            metric=metric,
                            alpha=alpha,
                            configs=inst.configs,
                        )
                        fried_rows.append(
                            {"family": inst.family, "group": inst.group, **fried},
                        )
                    out = results_dir / f"{dataset_name}_friedman_{_safe(metric)}.csv"
                    pd.DataFrame(fried_rows).to_csv(out, index=False)
                    n_sig = sum(1 for r in fried_rows if r["significant"])
                    logger.info(
                        "    friedman: %d/%d family instances significant -> %s",
                        n_sig,
                        len(fried_rows),
                        out,
                    )

                if instances:
                    pair_frames = [
                        pairwise_significance(
                            eval_df,
                            metric=metric,
                            alpha=alpha,
                            correction=correction,
                            include_effect_size=effect_size,
                            pairs=inst.pairs,
                            family=inst.family,
                            group=inst.group,
                            include_cohens_d=include_cohens_d,
                            diff_ci=bootstrap_enabled,
                            n_iterations=bootstrap_iters,
                        )
                        for inst in instances
                    ]
                    pairs_df = pd.concat(pair_frames, ignore_index=True)
                    out = results_dir / f"{dataset_name}_pairwise_{_safe(metric)}.csv"
                    pairs_df.to_csv(out, index=False)
                    logger.info(
                        "    pairwise (%s correction, per family): %d pairs across "
                        "%d family instances -> %s",
                        correction,
                        len(pairs_df),
                        len(instances),
                        out,
                    )
            else:
                logger.warning(
                    "    aggregated evaluation only — emitting comparative table "
                    "without inferential statistics (re-run step 06 with "
                    "evaluate_per_user to enable Wilcoxon/Friedman/bootstrap)."
                )
                comp = (
                    eval_df[["model_name", "embedding_name", metric]]
                    .copy()
                    .assign(
                        config=lambda df: df["model_name"] + "_" + df["embedding_name"],
                    )
                    .loc[:, ["config", metric]]
                )
                out = results_dir / f"{dataset_name}_aggregated_{_safe(metric)}.csv"
                comp.to_csv(out, index=False)
                logger.info("    aggregated table saved: %s", out)

    logger.info("Statistical analyses complete.")

    try:
        from src.reporting.consolidate import write_consolidated

        written = write_consolidated(results_dir)
        for label, path in written.items():
            logger.info("Long-format %s: %s", label, path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Long-format consolidation skipped: %s", exc)


def _load_evaluation(
    results_dir: Path,
    dataset_name: str,
    condition: str,
) -> pd.DataFrame | None:
    """Load the evaluation table for a dataset/condition pair.

    For ``condition == "all"`` we merge the frozen and finetuned tables
    and tag each row with its origin condition so downstream tests can
    compare batteries side-by-side.
    """
    if condition == "all":
        dfs: list[pd.DataFrame] = []
        for cond in ("frozen", "finetuned"):
            path = results_dir / f"{dataset_name}_evaluation_{cond}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["condition"] = cond
                dfs.append(df)
        if not dfs:
            logger.warning("  No results found for %s", dataset_name)
            return None
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(
            results_dir / f"{dataset_name}_evaluation_combined.csv",
            index=False,
        )
        return combined

    path = results_dir / f"{dataset_name}_evaluation_{condition}.csv"
    if not path.exists():
        logger.warning("  Evaluation file not found: %s", path)
        return None
    return pd.read_csv(path)


def _metrics_to_test(
    eval_df: pd.DataFrame,
    k_values: list[int],
    primary_metrics: list[str],
    include_derived: bool,
) -> list[str]:
    """Metric columns to analyse: primary families by default (C2).

    Under LOO, precision@k and map@k are deterministic transforms of
    recall@k and the hit rank — they are only included when
    ``include_derived_metrics`` is set, and must not be read as
    independent evidence.
    """
    metric_families = list(primary_metrics)
    if include_derived:
        metric_families += [m for m in ("precision", "map") if m not in metric_families]
    candidates: list[str] = []
    for k in k_values:
        candidates.extend(f"{fam}@{k}" for fam in metric_families)
    return [m for m in candidates if m in eval_df.columns]


def _safe(metric: str) -> str:
    """File-system-safe version of a metric name (``ndcg@10`` -> ``ndcg_at_10``)."""
    return metric.replace("@", "_at_")
