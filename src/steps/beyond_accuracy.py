"""Step 06b — Beyond-accuracy metrics (post-hoc, off the Evaluator hot path).

Consumes the per-user artifacts persisted at final evaluation
(``results/per_user/<dataset>/*.csv.gz`` — the ``top_items`` sufficient
statistic; rankings are NEVER recomputed) plus TRAIN-only statistics and
writes, per cut-off ``k``:

* ``efd@k`` — novelty (Mean Self-Information; Vargas & Castells 2011,
  eq. 14 reduced), from train-only item popularity.
* ``ild@k`` — visual diversity (Vargas & Castells 2011, eq. 16),
  computed in the FIXED reference space (native ResNet50) for every
  system — never in the evaluated extractor's own space.
* ``cat_entropy@k`` — Shannon entropy of top-k categories; only for
  datasets with ``expects_categories: true`` (Tradesy is explicitly
  N/A: no column is written, honouring the category contract).
* ``icov@k`` — item coverage.  AGGREGATE (one value per cell, not per
  user): the column is replicated across a cell's per-user rows for
  convenience and additionally written to
  ``{dataset}_beyond_accuracy_coverage.csv``, but it has NO per-user
  distribution — step 07 refuses to run per-user Wilcoxon/Friedman on
  it (see ``_metrics_to_test``); comparing systems on coverage needs a
  distinct statistical treatment (explicit decision required).

The new columns are merged into the SAME per-user tables the accuracy
metrics live in (``{dataset}_evaluation_{frozen|finetuned}.csv``), so
the statistical step treats ``efd``/``ild``/``cat_entropy`` like any
metric family once added to ``statistical.primary_metrics``.  The mean
tables are refreshed afterwards.

Leakage rule: item popularity and the coverage catalogue are derived
exclusively from ``train.csv`` — never from val/test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.beyond_accuracy import (
    catalog_coverage_at_k,
    compute_user_beyond_accuracy,
)
from src.evaluation.paired_loader import discover_cells
from src.evaluation.persistence import read_cell_artifact
from src.steps.evaluate import _route_targets, _write_mean_table
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: The persistence layer stores the first 20 ranked items per user;
#: cut-offs beyond that cannot be computed post-hoc.
MAX_PERSISTED_K = 20


def _train_popularity(train_df: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
    """``pop(i)`` = distinct TRAIN users of item ``i`` / total users.

    The denominator is the full user population (``user2idx``); under
    the leave-one-out split every user has train interactions, so this
    equals the train-user count.  Estimated on train only — using
    val/test would leak held-out popularity into the novelty metric.
    """
    pairs = train_df[["user_idx", "item_idx"]].drop_duplicates()
    counts = np.bincount(pairs["item_idx"].to_numpy(), minlength=n_items).astype(np.float64)
    return counts / float(n_users)


def _load_reference_embeddings(
    embeddings_dir: str, dataset_name: str, reference: str, n_items: int
) -> np.ndarray:
    """Load the fixed reference embedding matrix (fail loud if absent).

    Pre-registered decision: the SAME reference space (native ResNet50
    by default) scores ILD for every system, whatever extractor
    produced the recommendations — self-model diversity is not
    comparable across systems.
    """
    path = Path(embeddings_dir) / dataset_name / f"{reference}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"ILD reference embedding not found: {path}. The beyond_accuracy "
            f"step requires the fixed reference space ({reference!r}, "
            f"beyond_accuracy.reference_embedding) for every dataset; run "
            f"the extract step for it or point the config at an existing "
            f"native artifact."
        )
    embeddings = np.load(path, mmap_mode="r")
    if embeddings.shape[0] != n_items:
        raise ValueError(
            f"Reference embedding {path} has {embeddings.shape[0]} rows but the "
            f"dataset catalogue has {n_items} items; the artifact does not "
            f"match this dataset's item2idx."
        )
    return embeddings


def _load_categories(config: dict, dataset_name: str) -> np.ndarray | None:
    """Category array honouring the dataset contract (N/A stays N/A).

    ``expects_categories: false`` (Tradesy) → explicit N/A: return
    ``None`` without touching the provider — the metric is not
    computed and no ``cat_entropy@k`` column is written.
    ``expects_categories: true`` → the array must exist; a missing one
    is a contract violation and raises (fail-loud, mirroring
    :func:`src.data.categories.enforce_category_contract`).
    """
    contract = (config.get("dataset_contracts") or {}).get(dataset_name) or {}
    expects = contract.get("expects_categories")
    if expects is False:
        logger.info(
            "  %s: cat_entropy N/A by contract (expects_categories=false); "
            "no category column will be written.",
            dataset_name,
        )
        return None

    from src.data.categories import item_category_array

    categories = item_category_array(dataset_name, config["paths"]["data_processed"])
    if expects is True and categories is None:
        raise RuntimeError(
            f"{dataset_name!r} declares expects_categories=true but "
            f"item_category_array() returned None; cannot compute "
            f"cat_entropy. Fix the raw category taxonomy or the contract "
            f"before running beyond_accuracy."
        )
    if categories is None:
        logger.info(
            "  %s: no category labels available (no contract declared); cat_entropy skipped.",
            dataset_name,
        )
    return categories


def _cell_frames(
    cell_paths: list[Path],
    popularity: np.ndarray,
    embeddings: np.ndarray,
    categories: np.ndarray | None,
    n_catalog_items: int,
    k_values: list[int],
    use_rank_relevance: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user beyond-accuracy frame + aggregate coverage frame.

    Returns ``(per_user_df, coverage_df)`` where ``per_user_df`` has
    one row per (cell, user) keyed by ``model_name`` / ``embedding_name``
    / ``user_id``, and ``coverage_df`` one row per (cell, k).
    """
    user_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for path in cell_paths:
        metadata, records = read_cell_artifact(path)
        recommender = metadata["recommender"]
        visual_config = metadata["visual_config"]
        top_lists = list(records["top_items"])

        zero_pop = {
            int(i) for top in top_lists for i in top[: max(k_values)] if popularity[int(i)] <= 0.0
        }
        if zero_pop:
            logger.warning(
                "    %s/%s: %d recommended item(s) have zero TRAIN popularity; "
                "they are excluded from the EFD average (see efd_at_k docstring).",
                recommender,
                visual_config,
                len(zero_pop),
            )

        for user_idx, top_items in zip(records["user_idx"], top_lists, strict=True):
            row: dict = {
                "model_name": recommender,
                "embedding_name": visual_config,
                "user_id": int(user_idx),
            }
            row.update(
                compute_user_beyond_accuracy(
                    top_items,
                    popularity,
                    embeddings,
                    categories,
                    k_values,
                    use_rank_relevance=use_rank_relevance,
                )
            )
            user_rows.append(row)

        cell_rows = user_rows[-len(top_lists) :]
        for k in k_values:
            icov = catalog_coverage_at_k(top_lists, n_catalog_items, k)
            # Share of top-k slots excluded from EFD (zero train
            # popularity), averaged over the cell's users: EFD values of
            # two cells are only comparable when this share is similar.
            excluded = float(np.nanmean([r[f"efd_excluded_frac@{k}"] for r in cell_rows]))
            coverage_rows.append(
                {
                    "model_name": recommender,
                    "embedding_name": visual_config,
                    "k": k,
                    "icov": icov,
                    "n_catalog_items": n_catalog_items,
                    "efd_excluded_frac_mean": excluded,
                }
            )
            if excluded > 0.0:
                logger.warning(
                    "    %s/%s k=%d: %.2f%% of top-%d slots excluded from EFD "
                    "(zero TRAIN popularity); EFD is averaged over the rest.",
                    recommender,
                    visual_config,
                    k,
                    100.0 * excluded,
                    k,
                )
            # Replicated onto the per-user rows for the shared tables;
            # AGGREGATE — never a per-user signal (see module docstring).
            for row in cell_rows:
                row[f"icov@{k}"] = icov

    return pd.DataFrame(user_rows), pd.DataFrame(coverage_rows)


def _merge_into_table(table_path: Path, ba_frame: pd.DataFrame) -> None:
    """Merge the beyond-accuracy columns into an evaluation CSV in place.

    Idempotent: pre-existing beyond-accuracy columns are dropped and
    rewritten.  Rows whose cell has no per-user artifact keep NaN (a
    warning reports how many).  Written via temp file + rename so an
    interrupted merge never leaves a half-written table.
    """
    keys = ["model_name", "embedding_name", "user_id"]
    ba_columns = [c for c in ba_frame.columns if c not in keys]
    df = pd.read_csv(table_path)
    df = df.drop(columns=[c for c in ba_columns if c in df.columns])
    merged = df.merge(ba_frame, on=keys, how="left", validate="many_to_one")

    n_missing = int(merged[ba_columns[0]].isna().sum()) if ba_columns else 0
    if n_missing:
        logger.warning(
            "  %s: %d row(s) have no matching per-user artifact "
            "(beyond-accuracy columns left NaN for those cells).",
            table_path.name,
            n_missing,
        )

    tmp = table_path.with_suffix(table_path.suffix + ".tmp")
    merged.to_csv(tmp, index=False)
    tmp.rename(table_path)
    logger.info(
        "  merged %d beyond-accuracy column(s) into %s (%d rows).",
        len(ba_columns),
        table_path.name,
        len(merged),
    )


def run() -> None:
    """Compute EFD/ILD/iCov/cat_entropy post-hoc for every dataset.

    Reads the persisted ``top_items`` (never re-ranks), computes the
    beyond-accuracy metrics against train-only statistics and the fixed
    reference embedding, merges the columns into the per-user
    evaluation tables and refreshes the mean tables.
    """
    config = load_config()
    ba_cfg = config.get("beyond_accuracy") or {}
    if not ba_cfg.get("enabled", True):
        logger.info("beyond_accuracy step disabled in config; skipping.")
        return

    datasets = config.get("datasets", [])
    if not datasets:
        logger.info("beyond_accuracy step skipped: datasets list is empty.")
        return

    k_values = [k for k in config.get("k_values", [5, 10, 20]) if k <= MAX_PERSISTED_K]
    dropped = [k for k in config.get("k_values", [5, 10, 20]) if k > MAX_PERSISTED_K]
    if dropped:
        logger.warning(
            "k values %s exceed the persisted top-%d and are skipped for beyond-accuracy metrics.",
            dropped,
            MAX_PERSISTED_K,
        )
    if not k_values:
        logger.warning("No usable k values (<= %d); nothing to compute.", MAX_PERSISTED_K)
        return

    reference = ba_cfg.get("reference_embedding", "resnet50")
    use_rank_relevance = bool(ba_cfg.get("use_rank_relevance", False))
    seed = int(config.get("seed", 42))

    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]
    results_root = Path(config["paths"].get("results", "results"))
    tables_dir = results_root / "tables"

    for dataset_name in datasets:
        logger.info("=== Dataset: %s ===", dataset_name)
        cell_paths = discover_cells(results_root, dataset_name, seed)
        if not cell_paths:
            logger.warning(
                "  No per-user artifacts under %s for seed %d; run the "
                "evaluate step first (full_ranking protocol).",
                results_root / "per_user" / dataset_name,
                seed,
            )
            continue

        base = Path(processed_dir) / dataset_name
        train_df = pd.read_csv(base / "train.csv")
        with open(base / "user2idx.json", encoding="utf-8") as fh:
            n_users = len(json.load(fh))
        with open(base / "item2idx.json", encoding="utf-8") as fh:
            n_items = len(json.load(fh))

        popularity = _train_popularity(train_df, n_users, n_items)
        # Coverage denominator: distinct TRAIN items (the recommendable
        # catalogue the systems learned from), never test items.
        n_catalog_items = int(train_df["item_idx"].nunique())
        logger.info(
            "  train stats: %d users, %d catalogue items (of %d indexed), "
            "reference embedding=%s, rank_relevance=%s",
            n_users,
            n_catalog_items,
            n_items,
            reference,
            use_rank_relevance,
        )

        embeddings = _load_reference_embeddings(embeddings_dir, dataset_name, reference, n_items)
        categories = _load_categories(config, dataset_name)

        ba_frame, coverage = _cell_frames(
            cell_paths,
            popularity,
            embeddings,
            categories,
            n_catalog_items,
            k_values,
            use_rank_relevance,
        )
        logger.info(
            "  computed beyond-accuracy metrics for %d cell(s), %d user rows.",
            len(cell_paths),
            len(ba_frame),
        )

        coverage_path = tables_dir / f"{dataset_name}_beyond_accuracy_coverage.csv"
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(coverage_path, index=False)
        logger.info("  coverage (aggregate, per cell x k): %s", coverage_path)

        merged_targets: set[str] = set()
        for path in cell_paths:
            metadata, _ = read_cell_artifact(path)
            merged_targets.update(
                _route_targets(metadata["recommender"], metadata["visual_config"])
            )
        for target in sorted(merged_targets):
            table_path = tables_dir / f"{dataset_name}_evaluation_{target}.csv"
            if not table_path.exists():
                logger.warning("  evaluation table missing, skipped: %s", table_path)
                continue
            _merge_into_table(table_path, ba_frame)
            _write_mean_table(tables_dir, dataset_name, target)

    logger.info("Beyond-accuracy step complete.")
