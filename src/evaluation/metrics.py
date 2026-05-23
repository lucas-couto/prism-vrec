"""Recommendation evaluation metrics.

All functions operate on a single user's ranked list and ground-truth set.
To obtain system-level scores, compute per-user values and average across
users (handled by :class:`~src.evaluation.protocol.Evaluator`).
"""

from __future__ import annotations

import math


def precision_at_k(ranked_list: list, ground_truth: set, k: int) -> float:
    """Precision@K: fraction of top-K items that are relevant.

    Parameters
    ----------
    ranked_list:
        Ordered list of recommended item IDs (most relevant first).
    ground_truth:
        Set of relevant item IDs for the user.
    k:
        Cut-off position.

    Returns
    -------
    float
        Precision value in [0, 1].
    """
    top_k = ranked_list[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / k


def recall_at_k(ranked_list: list, ground_truth: set, k: int) -> float:
    """Recall@K: fraction of relevant items found in the top-K.

    In the leave-one-out setting (|ground_truth| == 1) this is equivalent
    to the hit rate: 1.0 if the single test item appears in the top-K,
    0.0 otherwise.

    Parameters
    ----------
    ranked_list:
        Ordered list of recommended item IDs.
    ground_truth:
        Set of relevant item IDs.
    k:
        Cut-off position.

    Returns
    -------
    float
        Recall value in [0, 1].
    """
    if not ground_truth:
        return 0.0
    top_k = ranked_list[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / len(ground_truth)


def f1_at_k(ranked_list: list, ground_truth: set, k: int) -> float:
    """F1@K: harmonic mean of Precision@K and Recall@K.

    Returns 0.0 when both precision and recall are zero.

    Parameters
    ----------
    ranked_list:
        Ordered list of recommended item IDs.
    ground_truth:
        Set of relevant item IDs.
    k:
        Cut-off position.

    Returns
    -------
    float
        F1 value in [0, 1].
    """
    p = precision_at_k(ranked_list, ground_truth, k)
    r = recall_at_k(ranked_list, ground_truth, k)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def map_at_k(ranked_list: list, ground_truth: set, k: int) -> float:
    """MAP@K: mean average precision at K.

    Computes Average Precision (AP) by accumulating precision at each
    relevant position within the top-K, then dividing by
    ``min(k, |ground_truth|)``.

    In the leave-one-out setting (|ground_truth| == 1) this reduces to
    the reciprocal rank (MRR@K): ``1 / rank`` if the test item appears
    in the top-K, 0 otherwise.

    Parameters
    ----------
    ranked_list:
        Ordered list of recommended item IDs.
    ground_truth:
        Set of relevant item IDs.
    k:
        Cut-off position.

    Returns
    -------
    float
        AP value in [0, 1].
    """
    if not ground_truth:
        return 0.0

    top_k = ranked_list[:k]
    hits = 0
    sum_precision = 0.0

    for i, item in enumerate(top_k):
        if item in ground_truth:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(k, len(ground_truth))


def ndcg_at_k(ranked_list: list, ground_truth: set, k: int) -> float:
    """NDCG@K: normalised discounted cumulative gain.

    Uses binary relevance (1 if relevant, 0 otherwise).  The ideal DCG
    is computed assuming all relevant items occupy the top positions.

    Parameters
    ----------
    ranked_list:
        Ordered list of recommended item IDs.
    ground_truth:
        Set of relevant item IDs.
    k:
        Cut-off position.

    Returns
    -------
    float
        NDCG value in [0, 1].
    """
    if not ground_truth:
        return 0.0

    top_k = ranked_list[:k]

    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

    n_relevant = min(k, len(ground_truth))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_all_metrics(
    ranked_list: list,
    ground_truth: set,
    k_values: list[int],
) -> dict:
    """Compute all metrics for every cut-off in *k_values*.

    Parameters
    ----------
    ranked_list:
        Ordered list of recommended item IDs.
    ground_truth:
        Set of relevant item IDs.
    k_values:
        List of cut-off positions (e.g. ``[5, 10, 20]``).

    Returns
    -------
    dict
        Flat dictionary with keys like ``'precision@5'``, ``'ndcg@10'``,
        etc.  Values are floats.
    """
    results: dict[str, float] = {}
    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(ranked_list, ground_truth, k)
        results[f"recall@{k}"] = recall_at_k(ranked_list, ground_truth, k)
        results[f"f1@{k}"] = f1_at_k(ranked_list, ground_truth, k)
        results[f"map@{k}"] = map_at_k(ranked_list, ground_truth, k)
        results[f"ndcg@{k}"] = ndcg_at_k(ranked_list, ground_truth, k)
    return results
