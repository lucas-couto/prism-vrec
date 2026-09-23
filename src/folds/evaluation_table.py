"""Build the per-user evaluation table the back half of the pipeline reads.

When ``folds.enabled`` is true the ``folds`` step REPLACES the
single-split ``evaluate`` as the producer of the per-user records, and
it has to produce everything ``evaluate`` produced -- not only the
``results/per_user/`` artifacts but also
``results/tables/<dataset>_evaluation_<condition>.csv``, which
``beyond_accuracy`` merges its columns into and ``statistical`` reads.
Without it the statistical step logged "Evaluation file not found" and
completed in under a second, having tested nothing (found 2026-09-09 on
the first real fold run).

Nothing is recomputed on a GPU: under leave-one-out the held-out rank is
a sufficient statistic for every accuracy metric at any cut-off (see
:mod:`src.evaluation.derive_metrics`), and the folds already persisted
that rank for every user.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.derive_metrics import per_user_metrics
from src.evaluation.persistence import read_cell_artifact
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Written by ``evaluate`` but NOT derivable from a fold artifact: they
#: describe the model instance, and a fold cell trains K of them.  The
#: columns are omitted rather than filled with a guess; declared in
#: ``docs/protocol.md``.
UNAVAILABLE_FROM_FOLDS = ("visual_input_dim", "n_trainable_params")

#: Metrics ``evaluate`` records per user, in its column order.
_METRICS = ("precision", "recall", "f1", "map", "ndcg")


def _cell_rows(records: pd.DataFrame, metadata: dict, k_values: list[int]) -> pd.DataFrame:
    ranks = records["rank"].to_numpy()
    out = pd.DataFrame({"user_id": records["user_idx"].to_numpy()})
    for k in k_values:
        derived = per_user_metrics(ranks, k)
        for name in _METRICS:
            out[f"{name}@{k}"] = derived[name]
    return out.assign(
        protocol=metadata.get("eval_protocol", "full_ranking"),
        dataset=metadata["dataset"],
        model_name=metadata["recommender"],
        embedding_name=metadata["visual_config"],
    )


def write_evaluation_table(
    results_dir: str | Path,
    dataset: str,
    condition: str,
    k_values: list[int],
) -> Path | None:
    """Write ``<dataset>_evaluation_<condition>.csv`` from the fold artifacts.

    :returns: the path written, or ``None`` when the dataset has no
        concatenated artifact yet (nothing to tabulate is not an error).
    """
    results_dir = Path(results_dir)
    cell_dir = results_dir / "per_user" / dataset
    artifacts = sorted(cell_dir.glob("*.csv.gz")) if cell_dir.is_dir() else []
    if not artifacts:
        logger.warning("No per-user artifact under %s; evaluation table not written.", cell_dir)
        return None

    frames = []
    for path in artifacts:
        metadata, records = read_cell_artifact(path)
        frames.append(_cell_rows(records, metadata, k_values))

    table = pd.concat(frames, ignore_index=True)
    out_dir = results_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dataset}_evaluation_{condition}.csv"
    table.to_csv(out, index=False)
    logger.info(
        "  %s: evaluation table written from %d fold cell(s), %d row(s) -> %s "
        "(without %s, which describe one model instance and a fold cell trains K)",
        dataset,
        len(frames),
        len(table),
        out.name,
        ", ".join(UNAVAILABLE_FROM_FOLDS),
    )
    return out
