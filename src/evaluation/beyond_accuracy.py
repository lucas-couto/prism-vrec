"""Beyond-accuracy metrics: novelty (EFD), diversity (ILD), coverage, entropy.

All per-user functions follow the contract of
:mod:`src.evaluation.metrics`: they operate on a single user's ranked
list (best first, already truncated to the persisted top-N) and return a
scalar.  System-level scores are the mean over users, computed by the
post-hoc step (:mod:`src.steps.beyond_accuracy`) — never by the online
:class:`~src.evaluation.protocol.Evaluator`, whose hot path stays
untouched.

Formula sources
---------------
* **EFD** — Vargas & Castells (2011), "Rank and relevance in novelty
  and diversity metrics for recommender systems", RecSys '11, eq. 14
  (DOI 10.1145/2043932.2043955).  The default here is the reduced form
  without rank discount or relevance weighting — Mean Self-Information:
  ``EFD@k(u) = (1/k) · Σ_{i ∈ top-k(u)} log2(1 / pop(i))``.
  Alignment references: Vargas (2014, SIGIR doctoral abstract) and
  Deldjoo et al. (2021), which report EFD/iCov in the same setting.
* **ILD** — Vargas & Castells (2011), eq. 16 in the rank/relevance-free
  form: mean pairwise distance over the top-k, with
  ``dist(i, j) = 1 − sim(i, j)`` for a similarity normalised to [0, 1].
* **Item coverage** — catalogue fraction reached by the union of the
  users' top-k lists (Deldjoo et al. 2021).  AGGREGATE: one value per
  system, no per-user distribution — it must NOT enter per-user
  paired tests (Wilcoxon/Friedman) alongside the other metrics.
* **Category entropy** — Shannon entropy (base 2) of the category
  distribution inside the top-k.

Leakage rule: every population/catalogue statistic consumed here must
be estimated on the TRAIN split only (Vargas & Castells 2011, §9 build
their discovery models on training data); the post-hoc step enforces
this when building the ``popularity`` array and the catalogue size.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

#: Exponential rank-discount base of Vargas & Castells (2011, §7) used
#: when ``use_rank_relevance=True`` in :func:`efd_at_k`.
RANK_DISCOUNT_BASE = 0.85


def efd_at_k(
    top_items: list[int],
    popularity: np.ndarray,
    k: int,
    *,
    use_rank_relevance: bool = False,
) -> float:
    """EFD@K (novelty): mean self-information of the top-K items.

    ``EFD@k(u) = (1/|valid|) · Σ_{i ∈ valid(u)} log2(1 / pop(i))`` — the
    Mean Self-Information reduction of Vargas & Castells (2011), eq. 14,
    where ``valid(u)`` are the top-k items with ``pop(i) > 0``.

    Items with ``pop(i) <= 0`` (never seen in TRAIN) carry infinite
    self-information; they are EXCLUDED from the average (the mean is
    taken over the remaining items) rather than floored with an
    arbitrary epsilon.  The denominator is therefore ``|valid|``, not
    ``k``, so EFD is only comparable across models that recommend a
    similar share of cold items: :func:`efd_excluded_frac_at_k` reports
    that share per user (``efd_excluded_frac@k``) and the step
    aggregates it per cell, so the comparability is checked, not assumed.

    Parameters
    ----------
    top_items:
        Ordered list of recommended item indices (best first).
    popularity:
        ``(n_items,)`` array with ``pop(i)`` in [0, 1]: fraction of
        users that interacted with item ``i`` in the TRAIN split.
        Estimating this on val/test would leak held-out information
        into the metric.
    k:
        Cut-off position.
    use_rank_relevance:
        Off by default (MSI form).  When True, applies the exponential
        rank discount ``disc(n) = 0.85^n`` (0-indexed rank ``n``,
        base :data:`RANK_DISCOUNT_BASE`) normalised by ``Σ disc(n)``,
        as in the full eq. 14 of Vargas & Castells (2011).  The
        relevance weight ``p(rel | i, u)`` of the full formula is NOT
        applied (treated as 1): under post-hoc leave-one-out there is
        no relevance model to estimate it from.  Kept only for future
        exact comparability with Deldjoo et al. (2021).

    Returns
    -------
    float
        EFD value (>= 0), or ``nan`` when no top-K item has positive
        train popularity (or the list is empty).
    """
    top_k = top_items[:k]
    if not top_k:
        return float("nan")

    pops = np.asarray(popularity, dtype=np.float64)[np.asarray(top_k, dtype=np.int64)]
    valid = pops > 0.0
    if not valid.any():
        return float("nan")

    self_information = -np.log2(pops[valid])
    if not use_rank_relevance:
        return float(self_information.mean())

    ranks = np.flatnonzero(valid)  # 0-indexed positions of the valid items
    discounts = RANK_DISCOUNT_BASE ** ranks.astype(np.float64)
    return float((discounts * self_information).sum() / discounts.sum())


def efd_excluded_frac_at_k(top_items: list[int], popularity: np.ndarray, k: int) -> float:
    """Share of the top-K items excluded from EFD@K (zero TRAIN popularity).

    Companion of :func:`efd_at_k`: ``|{i ∈ top-k : pop(i) <= 0}| / |top-k|``,
    ``nan`` for an empty list.  A model whose value is materially higher
    than another's has its EFD averaged over fewer terms, and the two
    EFD values are not directly comparable.
    """
    top_k = top_items[:k]
    if not top_k:
        return float("nan")
    pops = np.asarray(popularity, dtype=np.float64)[np.asarray(top_k, dtype=np.int64)]
    return float(np.mean(pops <= 0.0))


def ild_at_k(top_items: list[int], embeddings: np.ndarray, k: int) -> float:
    """ILD@K (diversity): mean pairwise visual distance in the top-K.

    ``ILD@k(u) = (2 / (k·(k-1))) · Σ_{i<j ∈ top-k(u)} dist(i, j)`` —
    Vargas & Castells (2011), eq. 16 without rank/relevance weighting.

    ``dist(i, j) = 1 − (cos_sim(i, j) + 1) / 2``: the cosine similarity
    is normalised to [0, 1] BEFORE taking the complement, because raw
    visual embeddings can produce negative cosines while the paper
    specifies the complement of a [0, 1]-normalised similarity.  The
    resulting distance is therefore guaranteed to lie in [0, 1].

    METHODOLOGICAL DECISION (pre-registered): ``embeddings`` must be
    the FIXED REFERENCE space — the native ResNet50 features — for
    every system under comparison, never the embedding of the extractor
    that produced the recommendations.  "Self-model" diversity is not
    comparable across systems: each extractor deforms its own space in
    its own favour.  The post-hoc step loads the reference from
    ``data/embeddings/<dataset>/resnet50.npy`` regardless of the cell's
    extractor.

    Parameters
    ----------
    top_items:
        Ordered list of recommended item indices.
    embeddings:
        ``(n_items, dim)`` reference embedding matrix; row ``i`` is
        item index ``i`` (the extract step writes rows in
        ``item2idx`` order).
    k:
        Cut-off position.

    Returns
    -------
    float
        ILD value in [0, 1], or ``nan`` when fewer than 2 items with a
        non-zero embedding are available (a 1-item list has undefined
        ILD — reported as missing, never forced to 0).
    """
    top_k = top_items[:k]
    vectors = np.asarray(embeddings)[np.asarray(top_k, dtype=np.int64)].astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 0.0]
    if vectors.shape[0] < 2:
        return float("nan")

    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    distance = 1.0 - (cosine + 1.0) / 2.0
    rows, cols = np.triu_indices(vectors.shape[0], k=1)
    return float(distance[rows, cols].mean())


def catalog_coverage_at_k(
    top_lists: Iterable[list[int]],
    n_catalog_items: int,
    k: int,
) -> float:
    """iCov@K (item coverage): catalogue fraction reached across users.

    ``iCov@k = |∪_u top-k(u)| / n_catalog_items`` (Deldjoo et al. 2021).

    AGGREGATE METRIC: one value per (dataset, system, k), with NO
    per-user distribution — it cannot enter the per-user Wilcoxon/
    Friedman families like the other metrics; comparing systems on
    coverage needs a distinct treatment (e.g. across seeds/datasets).

    ``n_catalog_items`` must be the number of distinct items in the
    TRAIN split (the recommendable catalogue the systems learned from),
    documented and computed by the caller — never from test data.

    Returns ``nan`` when the catalogue size is not positive.
    """
    if n_catalog_items <= 0:
        return float("nan")
    covered: set[int] = set()
    for top_items in top_lists:
        covered.update(int(i) for i in top_items[:k])
    return len(covered) / n_catalog_items


def category_entropy_at_k(
    top_items: list[int],
    item_categories: np.ndarray | None,
    k: int,
) -> float | None:
    """Shannon entropy (base 2) of the top-K category distribution.

    ``cat_entropy@k(u) = −Σ_c p(c) · log2 p(c)`` with ``p(c)`` the
    fraction of top-K items in category ``c``.

    ``item_categories`` is the array built by
    :func:`src.data.categories.item_category_array` (the same source
    DeepStyle/fine-tuning consume); its "unlabelled" bucket counts as
    a category of its own.  For datasets without categories (Tradesy,
    ``expects_categories: false``) the array is ``None`` and the metric
    is explicitly N/A: this function returns ``None`` and the caller
    logs the omission — it never invents categories.

    Returns
    -------
    float | None
        Entropy in ``[0, log2(k)]``; ``nan`` for an empty list;
        ``None`` when the dataset has no categories (N/A).
    """
    if item_categories is None:
        return None
    top_k = top_items[:k]
    if not top_k:
        return float("nan")

    labels = np.asarray(item_categories)[np.asarray(top_k, dtype=np.int64)]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(top_k)
    return float(-(probabilities * np.log2(probabilities)).sum())


def compute_user_beyond_accuracy(
    top_items: list[int],
    popularity: np.ndarray,
    embeddings: np.ndarray,
    item_categories: np.ndarray | None,
    k_values: list[int],
    *,
    use_rank_relevance: bool = False,
) -> dict[str, float]:
    """Per-user beyond-accuracy metrics for every cut-off in *k_values*.

    Additive companion of
    :func:`src.evaluation.metrics.compute_all_metrics` (whose signature
    is a contract and stays untouched).  Keys: ``efd@k``,
    ``efd_excluded_frac@k`` (share of the top-k left out of EFD for zero
    train popularity), ``ild@k`` and — only when the dataset ships
    categories — ``cat_entropy@k``.
    The aggregate iCov is NOT computed here (it has no per-user value);
    see :func:`catalog_coverage_at_k`.
    """
    results: dict[str, float] = {}
    for k in k_values:
        results[f"efd@{k}"] = efd_at_k(
            top_items, popularity, k, use_rank_relevance=use_rank_relevance
        )
        results[f"efd_excluded_frac@{k}"] = efd_excluded_frac_at_k(top_items, popularity, k)
        results[f"ild@{k}"] = ild_at_k(top_items, embeddings, k)
        entropy = category_entropy_at_k(top_items, item_categories, k)
        if entropy is not None:
            results[f"cat_entropy@{k}"] = entropy
    return results


__all__ = [
    "RANK_DISCOUNT_BASE",
    "catalog_coverage_at_k",
    "category_entropy_at_k",
    "compute_user_beyond_accuracy",
    "efd_at_k",
    "efd_excluded_frac_at_k",
    "ild_at_k",
]
