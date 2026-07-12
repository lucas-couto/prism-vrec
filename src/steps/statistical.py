"""Step 07 — Statistical reporting.

For every dataset and every metric configured in
``configs/evaluation.yaml`` this step produces three artefacts:

* ``{dataset}_summary_{metric}.csv``
  Per-model mean with bootstrap confidence intervals.

* ``{dataset}_friedman_{metric}.csv``
  Friedman omnibus test across all models — answers "is *anyone*
  different?" before pairwise testing.

* ``{dataset}_pairwise_{metric}.csv``
  All-pairs Wilcoxon signed-rank test with multiple-comparison
  correction (Holm by default) and effect sizes (Cohen's d + Cliff's
  delta).

When per-user metrics are not available (the evaluator was run with
aggregated outputs only), the step falls back to writing a comparative
table without inferential statistics and logs a warning.

Configuration knobs (``configs/evaluation.yaml`` -> ``statistical:``):

* ``alpha``                       — family-wise significance level (default 0.05)
* ``correction``                  — ``"holm"`` (default), ``"bonferroni"``, ``"none"``
* ``bootstrap.enabled``           — toggle bootstrap CIs (default true)
* ``bootstrap.n_iterations``      — number of resamples (default 1000)
* ``friedman.enabled``            — toggle Friedman test (default true)
* ``effect_size``                 — toggle effect-size columns (default true)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    bootstrap_cfg = stat_cfg.get("bootstrap", {})
    bootstrap_enabled = bootstrap_cfg.get("enabled", True)
    bootstrap_iters = bootstrap_cfg.get("n_iterations", 1000)
    friedman_enabled = stat_cfg.get("friedman", {}).get("enabled", True)
    effect_size = stat_cfg.get("effect_size", True)

    results_dir = Path(config.get("paths", {}).get("results", "results")) / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Condition: %s  alpha=%.3f  correction=%s  bootstrap=%s  friedman=%s  effect_size=%s",
        condition,
        alpha,
        correction,
        bootstrap_enabled,
        friedman_enabled,
        effect_size,
    )

    for dataset_name in datasets:
        logger.info("=== Dataset: %s ===", dataset_name)
        eval_df = _load_evaluation(results_dir, dataset_name, condition)
        if eval_df is None:
            continue

        metrics = _metrics_to_test(eval_df, k_values)
        if not metrics:
            logger.warning("  No supported metric columns found in evaluation file.")
            continue

        per_user = "user_id" in eval_df.columns

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

                if friedman_enabled:
                    fried = friedman_test(eval_df, metric=metric, alpha=alpha)
                    out = results_dir / f"{dataset_name}_friedman_{_safe(metric)}.csv"
                    pd.DataFrame([fried]).to_csv(out, index=False)
                    logger.info(
                        "    friedman: chi2=%.3f p=%.4g significant=%s -> %s",
                        fried["statistic"],
                        fried["p_value"],
                        fried["significant"],
                        out,
                    )

                pairs = pairwise_significance(
                    eval_df,
                    metric=metric,
                    alpha=alpha,
                    correction=correction,
                    include_effect_size=effect_size,
                )
                out = results_dir / f"{dataset_name}_pairwise_{_safe(metric)}.csv"
                pairs.to_csv(out, index=False)
                logger.info("    pairwise (%s correction) saved: %s", correction, out)
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


def _metrics_to_test(eval_df: pd.DataFrame, k_values: list[int]) -> list[str]:
    """Return the supported metric column names actually present in the table."""
    candidates: list[str] = []
    for k in k_values:
        candidates.extend([f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"map@{k}"])
    return [m for m in candidates if m in eval_df.columns]


def _safe(metric: str) -> str:
    """File-system-safe version of a metric name (``ndcg@10`` -> ``ndcg_at_10``)."""
    return metric.replace("@", "_at_")
