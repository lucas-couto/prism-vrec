"""Fusion strategies for combining multiple embedding matrices.

Each strategy receives a list of matrices ``[Z1, Z2, ..., ZM]`` where
``Zm`` has shape ``(N, dm)`` and returns a fused matrix ``H`` with shape
``(N, Dh)``.  Optional L2 normalisation can be applied per-vector before
fusion.

Eleven strategies are provided (ten offline plus the learned
``adaptive_gated``, which lives in :mod:`src.fusions.online`):

*   **mean** -- element-wise mean (equal dims required)
*   **sum** -- element-wise sum (equal dims required)
*   **prod** -- Hadamard product (equal dims required)
*   **max_pool** -- element-wise max (equal dims required)
*   **weighted_mean** -- fixed configurable per-source weights (equal dims)
*   **softmax_weighted** -- softmax over fixed configurable logits (equal dims)
*   **sigmoid_gated** -- normalised sigmoids over fixed configurable logits (equal dims)
*   **concat** -- concatenation along the feature axis (any dims)
*   **pca** -- PCA on the concatenation (any dims)
*   **pca_per_model** -- separate PCA per source, then concatenate (any dims)
*   **adaptive_gated** -- online per-item gate MLP co-trained with the
    recommender (equal dims; no offline artifact)

Use :func:`get_fusion_strategy` to obtain a strategy callable by name.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _warn_ignored_kwargs(strategy: str, kwargs: dict) -> None:
    """Warn when a strategy receives kwargs it does not consume.

    Every strategy accepts ``**kwargs`` (part of the plugin contract),
    which means a typo'd hyperparameter (``n_component=...``) or one
    aimed at a different strategy is silently discarded — an experiment
    can appear to sweep a value that never varied.  Warn loudly instead.
    """
    if kwargs:
        logger.warning(
            "Fusion strategy '%s' ignored unknown kwargs %s — "
            "check for typos in configs/fusion.yaml.",
            strategy,
            sorted(kwargs),
        )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalise each row of *x*.

    Rows with zero norm are left unchanged (avoiding division by zero).

    Parameters
    ----------
    x:
        2-D array of shape ``(N, d)``.

    Returns
    -------
    np.ndarray
        Array of the same shape with unit-norm rows.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero for zero-vectors.
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _validate_embeddings(embeddings: list[np.ndarray]) -> None:
    """Check that *embeddings* is a non-empty list of 2-D arrays with the
    same number of rows ``N``."""
    if not embeddings:
        raise ValueError("embeddings list must not be empty.")
    if not all(isinstance(e, np.ndarray) for e in embeddings):
        raise TypeError("All elements of embeddings must be numpy arrays.")
    if any(e.ndim != 2 for e in embeddings):
        raise ValueError("All embedding arrays must be 2-D.")
    n_rows = embeddings[0].shape[0]
    if any(e.shape[0] != n_rows for e in embeddings):
        raise ValueError(
            "All embedding arrays must have the same number of rows (N). "
            f"Got shapes: {[e.shape for e in embeddings]}"
        )


def _validate_equal_dims(embeddings: list[np.ndarray]) -> None:
    """Check that all arrays have the same feature dimensionality."""
    dims = {e.shape[1] for e in embeddings}
    if len(dims) > 1:
        raise ValueError(
            "This fusion strategy requires all embeddings to have the same "
            f"dimensionality, but got dimensions: {sorted(dims)}"
        )


def _maybe_normalize(embeddings: list[np.ndarray], normalize: bool) -> list[np.ndarray]:
    """Optionally L2-normalise every matrix in the list."""
    if normalize:
        return [l2_normalize(e) for e in embeddings]
    return embeddings


def fuse_mean(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Element-wise mean of embedding matrices.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d)`` with identical *d*.
    normalize:
        If ``True``, L2-normalise each matrix before fusion.

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, d)``.
    """
    _warn_ignored_kwargs("mean", kwargs)
    _validate_embeddings(embeddings)
    _validate_equal_dims(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)
    stacked = np.stack(embeddings, axis=0)  # (M, N, d)
    return np.mean(stacked, axis=0)


def fuse_sum(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Element-wise sum of embedding matrices.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d)`` with identical *d*.
    normalize:
        If ``True``, L2-normalise each matrix before fusion.

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, d)``.
    """
    _warn_ignored_kwargs("sum", kwargs)
    _validate_embeddings(embeddings)
    _validate_equal_dims(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)
    stacked = np.stack(embeddings, axis=0)  # (M, N, d)
    return np.sum(stacked, axis=0)


def fuse_prod(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Hadamard (element-wise) product of embedding matrices.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d)`` with identical *d*.
    normalize:
        If ``True``, L2-normalise each matrix before fusion.

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, d)``.
    """
    _warn_ignored_kwargs("prod", kwargs)
    _validate_embeddings(embeddings)
    _validate_equal_dims(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)
    stacked = np.stack(embeddings, axis=0)  # (M, N, d)
    return np.prod(stacked, axis=0)


def fuse_max_pool(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Element-wise max-pooling across embedding matrices.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d)`` with identical *d*.
    normalize:
        If ``True``, L2-normalise each matrix before fusion.

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, d)``.
    """
    _warn_ignored_kwargs("max_pool", kwargs)
    _validate_embeddings(embeddings)
    _validate_equal_dims(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)
    stacked = np.stack(embeddings, axis=0)  # (M, N, d)
    return np.max(stacked, axis=0)


def fuse_weighted_mean(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    *,
    weights: list[float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Weighted average of embedding matrices.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d)`` with identical *d*.
    normalize:
        If ``True``, L2-normalise each matrix before fusion.
    weights:
        Per-source weights ``[w1, w2, ..., wM]``.  They need not sum to 1
        -- they are normalised internally.  If ``None``, falls back to
        uniform weights (equivalent to plain mean).

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, d)``.
    """
    _warn_ignored_kwargs("weighted_mean", kwargs)
    _validate_embeddings(embeddings)
    _validate_equal_dims(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)

    m = len(embeddings)
    if weights is None:
        weights = [1.0 / m] * m
    else:
        if len(weights) != m:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of embedding sources ({m})."
            )
        total = sum(weights)
        if total == 0:
            raise ValueError("Weights must not sum to zero.")
        weights = [w / total for w in weights]

    # Accumulate in float64 for precision, emit the pipeline's float32
    # (every feature artifact is validated against FEATURE_DTYPE).
    result = np.zeros_like(embeddings[0], dtype=np.float64)
    for w, emb in zip(weights, embeddings, strict=False):
        result += w * emb
    return result.astype(np.float32, copy=False)


def fuse_softmax_weighted(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    *,
    logits: list[float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Attention-weighted fusion via softmax over fixed configurable logits.

    Each source *m* receives a scalar attention weight
    ``alpha_m = softmax(logits)[m]``.  The fused representation is the
    weighted sum over sources.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d)`` with identical *d*.
    normalize:
        If ``True``, L2-normalise each matrix before fusion.
    logits:
        Raw (un-normalised) attention logits, one per source.  If ``None``,
        all logits default to 0 (uniform attention).

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, d)``.
    """
    _warn_ignored_kwargs("softmax_weighted", kwargs)
    _validate_embeddings(embeddings)
    _validate_equal_dims(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)

    m = len(embeddings)
    if logits is None:
        logits = [0.0] * m
    else:
        if len(logits) != m:
            raise ValueError(
                f"Number of logits ({len(logits)}) must match number of embedding sources ({m})."
            )

    # Subtract max for numerical stability.
    logits_arr = np.array(logits, dtype=np.float64)
    logits_arr -= logits_arr.max()
    exp_logits = np.exp(logits_arr)
    alphas = exp_logits / exp_logits.sum()

    result = np.zeros_like(embeddings[0], dtype=np.float64)
    for alpha, emb in zip(alphas, embeddings, strict=False):
        result += alpha * emb
    return result.astype(np.float32, copy=False)


def fuse_sigmoid_gated(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    *,
    logits: list[float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Gated fusion via normalised sigmoids over fixed configurable logits.

    Each source *m* receives a gate value ``g_m = sigmoid(logit_m)``.  The
    gates are then normalised to sum to 1 so that the fused representation
    is a convex combination of the sources.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d)`` with identical *d*.
    normalize:
        If ``True``, L2-normalise each matrix before fusion.
    logits:
        Raw logits passed through sigmoid then normalised.  If ``None``,
        all logits default to 0 (uniform gating, since sigmoid(0) = 0.5
        for all sources).

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, d)``.
    """
    _warn_ignored_kwargs("sigmoid_gated", kwargs)
    _validate_embeddings(embeddings)
    _validate_equal_dims(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)

    m = len(embeddings)
    if logits is None:
        logits = [0.0] * m
    else:
        if len(logits) != m:
            raise ValueError(
                f"Number of logits ({len(logits)}) must match number of embedding sources ({m})."
            )

    logits_arr = np.array(logits, dtype=np.float64)
    gates = 1.0 / (1.0 + np.exp(-logits_arr))
    gate_sum = gates.sum()
    if gate_sum == 0:
        raise ValueError("All gates are zero; cannot normalise.")
    gates = gates / gate_sum

    result = np.zeros_like(embeddings[0], dtype=np.float64)
    for gate, emb in zip(gates, embeddings, strict=False):
        result += gate * emb
    return result.astype(np.float32, copy=False)


def fuse_concat(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Concatenation along the feature axis.

    This strategy accepts embeddings with different dimensionalities.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d_m)`` (dims may differ).
    normalize:
        If ``True``, L2-normalise each matrix before fusion.

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, sum(d_m))``.
    """
    _warn_ignored_kwargs("concat", kwargs)
    _validate_embeddings(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)
    return np.concatenate(embeddings, axis=1)


def _fit_pca_train_only(
    matrix: np.ndarray,
    n_components: int,
    random_state: int | None,
    train_items: np.ndarray | None,
    label: str,
    allow_transductive: bool = False,
) -> np.ndarray:
    """Fit PCA on train-item rows only, transform every row.

    Fitting on the full item matrix would leak information from items
    that only appear in validation/test interactions into the learned
    components.  ``train_items`` is the array of item indices with at
    least one *training* interaction.

    ``train_items=None`` raises unless ``allow_transductive=True`` is
    passed explicitly: a transductive fit over all rows is the exact
    test→fit leak the v2 protocol eliminated, so it must be opted into
    (intended for synthetic unit tests), never reached by accident.
    Logs the cumulative explained variance for the chosen ``k``.
    """
    n_components = min(n_components, *matrix.shape)
    if train_items is None:
        if not allow_transductive:
            raise ValueError(
                f"{label}: train_items is None. A transductive PCA fit over "
                "ALL rows leaks validation/test-only item structure into the "
                "components — the exact leak the v2 protocol eliminated. Pass "
                "the train item indices, or set allow_transductive=True to opt "
                "into the fit-on-all-rows behaviour explicitly (synthetic unit "
                "tests only)."
            )
        logger.warning(
            "%s: no train_items provided — PCA fit on ALL rows "
            "(transductive, explicitly opted in). Pass train item indices "
            "for a train-only fit.",
            label,
        )
        fit_rows = matrix
        # ``copy=True``: ``fit_rows`` IS ``matrix`` here, and centring it
        # in place would make the ``transform`` below subtract the mean
        # from already-centred data.
        pca = fit_pca_on_rows(fit_rows, n_components, random_state, label, copy=True)
    else:
        fit_rows = matrix[np.asarray(train_items)]
        n_components = min(n_components, *fit_rows.shape)
        pca = fit_pca_on_rows(fit_rows, n_components, random_state, label, copy=True)
    return pca.transform(matrix)


def fit_pca_on_rows(
    fit_rows: np.ndarray,
    n_components: int,
    random_state: int | None,
    label: str,
    *,
    copy: bool = True,
) -> PCA:
    """Fit a :class:`~sklearn.decomposition.PCA` on *fit_rows* and log it.

    Factored out of :func:`_fit_pca_train_only` so the streaming
    executor (:mod:`src.fusions.streaming`) fits the *same* estimator on
    the *same* rows — identical components, identical outputs — without
    having to materialise the full matrix it will later transform.

    ``copy=False`` lets scikit-learn centre *fit_rows* in place, halving
    the peak of the fit.  Pass it only when the caller owns *fit_rows*
    and will not read it again afterwards.

    :param fit_rows:
        The rows the components are fit on, shape ``(n_fit, D)``.
    :param n_components:
        Component count, already clamped to ``min(k, n_fit, D)`` by the
        caller.
    :param random_state:
        Seed forwarded to scikit-learn, so the randomized solver is
        reproducible.
    :param label:
        Prefix for the log line identifying the strategy and source.
    :param copy:
        Forwarded to :class:`~sklearn.decomposition.PCA`.
    :returns:
        The fitted estimator.
    """
    pca = PCA(n_components=n_components, random_state=random_state, copy=copy)
    pca.fit(fit_rows)
    explained = float(np.sum(pca.explained_variance_ratio_))
    logger.info(
        "%s: PCA k=%d fit on %d rows, cumulative explained variance %.4f",
        label,
        n_components,
        fit_rows.shape[0],
        explained,
    )
    return pca


def pca_align(
    embeddings: list[np.ndarray],
    dim: int,
    *,
    train_items: np.ndarray | None = None,
    random_state: int | None = 42,
    allow_transductive: bool = False,
) -> list[np.ndarray]:
    """Align sources of differing native dims to *dim* via per-source PCA.

    The PCA alignment used by the element-wise fusion family when
    ``alignment.method = pca``: each native matrix is independently
    reduced to ``dim`` (fit on train items only), so equal-dim
    operations become applicable.  The learned counterpart is
    :class:`src.fusions.online.LearnedAlignmentFusion`.

    ``allow_transductive`` is forwarded to :func:`_fit_pca_train_only`;
    leave it ``False`` in every production path (see that function).
    """
    return [
        _fit_pca_train_only(
            emb, dim, random_state, train_items, f"pca_align[src{i}]", allow_transductive
        )
        for i, emb in enumerate(embeddings)
    ]


def fuse_pca(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    *,
    n_components: int = 128,
    random_state: int | None = 42,
    train_items: np.ndarray | None = None,
    allow_transductive: bool = False,
    **kwargs,
) -> np.ndarray:
    """PCA on the concatenation of all embeddings.

    First concatenates along the feature axis, then applies PCA to reduce
    dimensionality to *n_components*.  The PCA is fit ONLY on rows of
    items with at least one training interaction (``train_items``) and
    applied to every row — fitting on all items would leak test-item
    structure into the components.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d_m)`` (dims may differ).
    normalize:
        If ``True``, L2-normalise each matrix before fusion.
    n_components:
        Number of principal components to keep.
    random_state:
        Seed for reproducibility.
    train_items:
        Indices of items appearing in the training split; the PCA fit
        set.
    allow_transductive:
        Opt-in escape hatch for ``train_items=None`` (fit on all rows).
        Leave ``False`` in production; ``True`` is for synthetic tests
        only (see :func:`_fit_pca_train_only`).

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, n_components)``.
    """
    _warn_ignored_kwargs("pca", kwargs)
    _validate_embeddings(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)
    concatenated = np.concatenate(embeddings, axis=1)

    return _fit_pca_train_only(
        concatenated, n_components, random_state, train_items, "pca", allow_transductive
    )


def fuse_pca_per_model(
    embeddings: list[np.ndarray],
    normalize: bool = True,
    *,
    n_components: int = 64,
    random_state: int | None = 42,
    train_items: np.ndarray | None = None,
    allow_transductive: bool = False,
    **kwargs,
) -> np.ndarray:
    """Separate PCA per source, then **concatenate**.

    Each embedding matrix is independently reduced to *n_components*
    dimensions via PCA (fit on train items only, see :func:`fuse_pca`).
    The reduced representations are then concatenated — NOT combined
    element-wise — yielding a final dimensionality of
    ``M * n_components``.

    Parameters
    ----------
    embeddings:
        List of arrays, each ``(N, d_m)`` (dims may differ).
    normalize:
        If ``True``, L2-normalise each matrix before fusion.
    n_components:
        Number of components to keep per source.
    random_state:
        Seed for reproducibility.
    train_items:
        Indices of items appearing in the training split; the PCA fit
        set.
    allow_transductive:
        Opt-in escape hatch for ``train_items=None`` (fit on all rows).
        Leave ``False`` in production; ``True`` is for synthetic tests
        only (see :func:`_fit_pca_train_only`).

    Returns
    -------
    np.ndarray
        Fused matrix of shape ``(N, M * n_components)``.
    """
    _warn_ignored_kwargs("pca_per_model", kwargs)
    _validate_embeddings(embeddings)
    embeddings = _maybe_normalize(embeddings, normalize)

    reduced = [
        _fit_pca_train_only(
            emb,
            n_components,
            random_state,
            train_items,
            f"pca_per_model[src{i}]",
            allow_transductive,
        )
        for i, emb in enumerate(embeddings)
    ]
    return np.concatenate(reduced, axis=1)


from src.fusions.registry import (  # noqa: E402 — placed here to keep the file flow readable
    get_fusion_strategy,  # re-exported for backwards compatibility
    register_fusion_strategy,
)


def _expand_weighted_mean(cfg: dict) -> list[tuple[str, dict]]:
    """Expand the ``w_cnn`` grid into ``[(suffix, {weights: [w, 1-w]})]``."""
    values = cfg.get("w_cnn", [0.5])
    if not isinstance(values, list):
        values = [values]
    return [(f"_w{w}", {"weights": [w, 1.0 - w]}) for w in values]


def _expand_fixed_logits(cfg: dict) -> list[tuple[str, dict]]:
    """Expand the ``logits`` grid into ``[(suffix, {logits: [...]})]``.

    Each grid entry is one full logit vector (one value per source),
    e.g. ``logits: [[1.0, 0.0]]``.  A bare vector is promoted to a
    single-entry grid.  Defaults to the uniform two-source vector
    (all-zero logits), which reduces to plain ``mean``.
    """
    values = cfg.get("logits", [[0.0, 0.0]])
    if values and not isinstance(values[0], list):
        values = [values]
    return [
        ("_l" + "_".join(f"{float(v):g}" for v in vec), {"logits": [float(v) for v in vec]})
        for vec in values
    ]


def _expand_pca(cfg: dict) -> list[tuple[str, dict]]:
    """Expand the ``n_components`` grid for the joint-PCA strategy."""
    values = cfg.get("n_components", [128])
    if not isinstance(values, list):
        values = [values]
    return [(f"_nc{n}", {"n_components": n}) for n in values]


def _expand_pca_per_model(cfg: dict) -> list[tuple[str, dict]]:
    """Expand the ``n_components_per_model`` grid for per-source PCA.

    The YAML key is ``n_components_per_model`` (it reads better next to
    ``pca``'s joint ``n_components``), but the kwarg handed to
    :func:`fuse_pca_per_model` must be the one that function actually
    consumes — ``n_components``.  Emitting the YAML spelling instead
    left it in ``**kwargs``, where ``_warn_ignored_kwargs`` discarded it
    and every task silently fell back to the default 64: a configured
    sweep of ``[32, 64, 128]`` wrote three identical 64-dim matrices
    under three different ``_nc`` filenames.
    """
    values = cfg.get("n_components_per_model", [64])
    if not isinstance(values, list):
        values = [values]
    return [(f"_nc{n}", {"n_components": n}) for n in values]


register_fusion_strategy("mean", fuse_mean, equal_dim_required=True)
register_fusion_strategy("sum", fuse_sum, equal_dim_required=True)
register_fusion_strategy("prod", fuse_prod, equal_dim_required=True)
register_fusion_strategy("max_pool", fuse_max_pool, equal_dim_required=True)
register_fusion_strategy(
    "weighted_mean",
    fuse_weighted_mean,
    equal_dim_required=True,
    expand_grid=_expand_weighted_mean,
)
register_fusion_strategy(
    "softmax_weighted",
    fuse_softmax_weighted,
    equal_dim_required=True,
    expand_grid=_expand_fixed_logits,
)
register_fusion_strategy(
    "sigmoid_gated",
    fuse_sigmoid_gated,
    equal_dim_required=True,
    expand_grid=_expand_fixed_logits,
)

register_fusion_strategy("concat", fuse_concat, equal_dim_required=False)
register_fusion_strategy(
    "pca",
    fuse_pca,
    equal_dim_required=False,
    expand_grid=_expand_pca,
)
register_fusion_strategy(
    "pca_per_model",
    fuse_pca_per_model,
    equal_dim_required=False,
    expand_grid=_expand_pca_per_model,
)


# Online (learned) strategies — fusion happens at training time as part
# of the recommender's forward pass; no offline .npy is produced.  The
# placeholder fn raises if invoked, since the offline pipeline must
# never call online strategies as functions.
def _online_placeholder(*_args, **_kwargs):  # noqa: ANN001
    raise NotImplementedError(
        "Online fusion strategies (e.g. 'adaptive_gated') are applied "
        "at training time as torch.nn.Module submodules of the "
        "recommender, not as offline numpy operations.  See "
        "src/fusions/online.py for the trainable implementation.",
    )


register_fusion_strategy(
    "adaptive_gated",
    _online_placeholder,
    equal_dim_required=True,
    online=True,
)


__all__ = [
    "fuse_mean",
    "fuse_sum",
    "fuse_prod",
    "fuse_max_pool",
    "fuse_weighted_mean",
    "fuse_softmax_weighted",
    "fuse_sigmoid_gated",
    "fuse_concat",
    "fuse_pca",
    "fuse_pca_per_model",
    "get_fusion_strategy",  # back-compat
    "l2_normalize",
]
