"""Derive accuracy metrics from held-out ranks (Task F).

Under leave-one-out (exactly one relevant item per user) the 1-indexed
rank ``r`` of the held-out in the masked, tie-broken ranking is a
*sufficient statistic* for every accuracy metric at any cut-off ``k``:

* HitRate@k  = 1[r <= k]
* Precision@k = HitRate@k / k
* Recall@k   = HitRate@k                (single relevant item)
* F1@k       = 2/(k+1) * HitRate@k
* MAP@k = MRR@k = (1/r) * 1[r <= k]
* NDCG@k     = 1[r <= k] / log2(r + 1)   (IDCG = 1)

So persisting the rank makes any metric recomputable forever without a
GPU.  These identities were verified against the online Evaluator in the
audit; the equivalence test locks the two implementations together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_METRICS = ("hitrate", "precision", "recall", "f1", "map", "mrr", "ndcg")


def per_user_metrics(ranks: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Return each metric at ``k`` as a per-user array, from 1-indexed ranks."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    ranks = np.asarray(ranks)
    if np.any(ranks < 1):
        raise ValueError("ranks must be 1-indexed (>= 1).")
    hit = (ranks <= k).astype(np.float64)
    inv_rank = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return {
        "hitrate": hit,
        "precision": hit / k,
        "recall": hit,
        "f1": hit * (2.0 / (k + 1)),
        "map": inv_rank,
        "mrr": inv_rank,
        "ndcg": np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0),
    }


def metrics_frame(records: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Per-user metrics DataFrame (``user_id`` + ``<metric>@<k>``) from records.

    ``records`` must have ``user_id`` and ``rank`` columns.
    """
    ranks = records["rank"].to_numpy()
    out = {"user_id": records["user_id"].to_numpy()}
    for k in k_values:
        for name, values in per_user_metrics(ranks, k).items():
            out[f"{name}@{k}"] = values
    return pd.DataFrame(out)


def aggregate(records: pd.DataFrame, k_values: list[int]) -> dict[str, float]:
    """Mean of every ``<metric>@<k>`` over users (matches Evaluator.evaluate)."""
    frame = metrics_frame(records, k_values)
    return {c: float(frame[c].mean()) for c in frame.columns if c != "user_id"}
