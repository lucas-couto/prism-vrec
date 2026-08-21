"""Chunked, memory-mapped execution of the offline fusion strategies.

The in-memory strategies in :mod:`src.fusions.strategies` hold every
source matrix, an L2-normalised copy of each, and the fused result at
once.  For the reference catalogues that is ~11.5 GB for a single
``concat`` (``amazon_women``: resnet50 2.85 GB + vit_b16 0.99 GB, their
normalised copies, and a 3.84 GB output), which is what let a pool of
fusion workers exhaust host memory.

This module executes the same strategies without ever materialising a
full matrix: sources are opened with ``mmap_mode="r"``, rows are
processed in chunks, and the result is written straight into a
memory-mapped ``.npy``.  Peak memory becomes a function of the chunk
size instead of the catalogue size — a few hundred MB rather than tens
of GB.

**The numbers do not change.**  L2 normalisation and concatenation are
row-wise, so a chunked pass produces bit-identical output.  PCA is
fitted on exactly the same matrix (the normalised, concatenated rows of
the training items, assembled once) via the shared
:func:`src.fusions.strategies.fit_pca_on_rows`, so the components are
identical; only the ``transform`` is chunked, and each output row
depends solely on its own input row.

Only the strategies that are row-wise decomposable live here.  Anything
else — including third-party strategies from ``plugins/fusions/`` —
keeps running through the registry's in-memory path, which remains the
contract every strategy is written against.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.fusions.strategies import (
    _validate_embeddings,
    _warn_ignored_kwargs,
    fit_pca_on_rows,
    l2_normalize,
)
from src.utils.atomic_io import atomic_np_memmap_save
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Rows processed per chunk.  At 8192 rows a two-source concat holds
#: ~200 MB (sources + normalised copies + output slice) regardless of
#: how many items the catalogue has.  Large enough that the BLAS calls
#: in ``PCA.transform`` stay efficient, small enough that the peak is
#: bounded by a constant.
CHUNK_ROWS = 8192

#: Strategies this module can execute row-wise.  Everything else falls
#: back to the in-memory registry path.
_STREAMABLE = frozenset({"concat", "pca", "pca_per_model"})

#: Component count each PCA strategy falls back to when the caller does
#: not pass one.  These MUST mirror the defaults in the signatures of
#: ``strategies.fuse_pca`` / ``strategies.fuse_pca_per_model``: the two
#: differ (128 vs 64), so a single shared default would silently change
#: the fused dimensionality of whichever strategy it did not match.
_DEFAULT_COMPONENTS = {"pca": 128, "pca_per_model": 64}


def is_streamable(strategy_name: str) -> bool:
    """Return ``True`` when *strategy_name* has a chunked implementation."""
    return strategy_name in _STREAMABLE


def run_streamed(
    strategy_name: str,
    emb_list_paths: list[str],
    output_path: str | Path,
    *,
    normalize: bool,
    train_items: np.ndarray | None = None,
    n_components: int | None = None,
    random_state: int | None = 42,
    chunk_rows: int = CHUNK_ROWS,
    **kwargs,
) -> tuple[int, int]:
    """Execute a streamable fusion, writing ``output_path`` directly.

    :param strategy_name:
        One of the names :func:`is_streamable` accepts.
    :param emb_list_paths:
        Source ``.npy`` paths, opened read-only and memory-mapped.
    :param output_path:
        Destination ``.npy``.  Written atomically: either the previous
        file or the complete new one.
    :param normalize:
        L2-normalise each source row before fusing, matching
        ``normalize_before_fusion`` in ``configs/fusion.yaml``.
    :param train_items:
        Item indices with at least one training interaction, the PCA fit
        set.  Required by the PCA strategies; ignored by ``concat``.
    :param n_components:
        Components kept — in total for ``pca``, per source for
        ``pca_per_model``.  ``None`` takes the same per-strategy default
        the in-memory function declares (128 and 64 respectively).
    :param random_state:
        Seed forwarded to scikit-learn.
    :param chunk_rows:
        Rows per pass.  Lower it to trade throughput for a smaller peak.
    :param kwargs:
        Unconsumed keyword arguments, warned about and discarded exactly
        as the in-memory strategies do (part of the plugin contract, and
        the reason a typo'd hyperparameter is loud rather than silent).
    :returns:
        The shape of the array written.
    :raises ValueError:
        If *strategy_name* is not streamable, or a PCA strategy is asked
        to run without *train_items* (the transductive fit is an opt-in
        the streaming path deliberately does not offer).
    """
    if not is_streamable(strategy_name):
        raise ValueError(
            f"{strategy_name!r} has no streaming implementation; "
            f"streamable strategies are {sorted(_STREAMABLE)}."
        )
    _warn_ignored_kwargs(strategy_name, kwargs)

    sources = [np.load(path, mmap_mode="r") for path in emb_list_paths]
    _validate_embeddings(sources)

    if strategy_name == "concat":
        return _stream_concat(sources, output_path, normalize, chunk_rows)

    if train_items is None:
        raise ValueError(
            f"{strategy_name}: train_items is required. A transductive PCA "
            "fit over ALL rows leaks validation/test-only item structure "
            "into the components; the streaming path does not offer the "
            "opt-in escape hatch (see strategies._fit_pca_train_only)."
        )
    fit_idx = np.asarray(train_items)
    if n_components is None:
        n_components = _DEFAULT_COMPONENTS[strategy_name]

    if strategy_name == "pca":
        return _stream_pca(
            sources, output_path, normalize, chunk_rows, fit_idx, n_components, random_state
        )
    return _stream_pca_per_model(
        sources, output_path, normalize, chunk_rows, fit_idx, n_components, random_state
    )


# ---------------------------------------------------------------------
# Row-wise building blocks
# ---------------------------------------------------------------------


def _prepare(source: np.ndarray, rows: slice | np.ndarray, normalize: bool) -> np.ndarray:
    """Materialise *rows* of *source* the way the in-memory path would.

    Fancy-indexing or slicing a memmap yields an ordinary array holding
    only those rows; :func:`l2_normalize` is then the *same* function
    the in-memory strategies call, so the values are identical.
    """
    block = np.asarray(source[rows])
    return l2_normalize(block) if normalize else block


def _concat_rows(
    sources: list[np.ndarray],
    rows: slice | np.ndarray,
    normalize: bool,
) -> np.ndarray:
    """Normalised, concatenated view of *rows* across every source."""
    return np.concatenate([_prepare(src, rows, normalize) for src in sources], axis=1)


def _chunks(n_rows: int, chunk_rows: int) -> Iterator[slice]:
    for start in range(0, n_rows, chunk_rows):
        yield slice(start, min(start + chunk_rows, n_rows))


def _output_dtype(sources: list[np.ndarray], normalize: bool) -> np.dtype:
    """dtype the in-memory path would produce, probed on a single row.

    Cheaper and more faithful than re-deriving numpy's promotion rules:
    it runs the real code on one row.  Guards against an empty
    catalogue, where there is no row to probe.
    """
    if sources[0].shape[0] == 0:
        return np.result_type(*[src.dtype for src in sources])
    return _concat_rows(sources, slice(0, 1), normalize).dtype


# ---------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------


def _stream_concat(
    sources: list[np.ndarray],
    output_path: str | Path,
    normalize: bool,
    chunk_rows: int,
) -> tuple[int, int]:
    """Chunked ``np.concatenate`` along the feature axis (bit-identical)."""
    n_rows = sources[0].shape[0]
    total_dim = sum(int(src.shape[1]) for src in sources)
    shape = (n_rows, total_dim)

    def _fill(out: np.memmap) -> None:
        for rows in _chunks(n_rows, chunk_rows):
            out[rows] = _concat_rows(sources, rows, normalize)

    atomic_np_memmap_save(
        output_path,
        dtype=_output_dtype(sources, normalize),
        shape=shape,
        fill=_fill,
    )
    _log_done("concat", shape, chunk_rows)
    return shape


def _stream_pca(
    sources: list[np.ndarray],
    output_path: str | Path,
    normalize: bool,
    chunk_rows: int,
    fit_idx: np.ndarray,
    n_components: int,
    random_state: int | None,
) -> tuple[int, int]:
    """PCA over the concatenation of all sources, fit on train rows only.

    The fit matrix is the one irreducible allocation: an exact PCA needs
    every training row at once.  It is assembled source by source so
    only one source block is held on top of it, and handed to
    scikit-learn with ``copy=False`` so the centring happens in place
    instead of doubling the peak.
    """
    n_rows = sources[0].shape[0]
    total_dim = sum(int(src.shape[1]) for src in sources)
    input_dtype = _output_dtype(sources, normalize)

    fit_rows = _assemble_fit_matrix(sources, fit_idx, normalize, total_dim, input_dtype)
    # Mirrors _fit_pca_train_only: clamped against the full matrix
    # first, then against the (smaller) fit matrix.
    k = min(n_components, n_rows, total_dim)
    k = min(k, *fit_rows.shape)
    pca = fit_pca_on_rows(fit_rows, k, random_state, "pca", copy=False)
    del fit_rows

    shape = (n_rows, int(pca.n_components_))
    # The dtype of a *transform*, not of the input: scikit-learn decides
    # it, so probe one row rather than predict it.
    dtype = pca.transform(_concat_rows(sources, slice(0, min(1, n_rows)), normalize)).dtype

    def _fill(out: np.memmap) -> None:
        for rows in _chunks(n_rows, chunk_rows):
            out[rows] = pca.transform(_concat_rows(sources, rows, normalize))

    atomic_np_memmap_save(output_path, dtype=dtype, shape=shape, fill=_fill)
    _log_done("pca", shape, chunk_rows)
    return shape


def _stream_pca_per_model(
    sources: list[np.ndarray],
    output_path: str | Path,
    normalize: bool,
    chunk_rows: int,
    fit_idx: np.ndarray,
    n_components: int,
    random_state: int | None,
) -> tuple[int, int]:
    """Independent PCA per source, then concatenation of the reductions."""
    n_rows = sources[0].shape[0]

    fitted = []
    for i, src in enumerate(sources):
        d = int(src.shape[1])
        block = _prepare(src, fit_idx, normalize)
        k = min(n_components, n_rows, d)
        k = min(k, *block.shape)
        fitted.append(fit_pca_on_rows(block, k, random_state, f"pca_per_model[src{i}]", copy=False))
        del block

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for pca in fitted:
        width = int(pca.n_components_)
        offsets.append((cursor, cursor + width))
        cursor += width
    shape = (n_rows, cursor)
    probe = slice(0, min(1, n_rows))
    dtype = np.result_type(
        *[
            pca.transform(_prepare(src, probe, normalize)).dtype
            for src, pca in zip(sources, fitted, strict=True)
        ]
    )

    def _fill(out: np.memmap) -> None:
        for rows in _chunks(n_rows, chunk_rows):
            for src, pca, (lo, hi) in zip(sources, fitted, offsets, strict=True):
                out[rows, lo:hi] = pca.transform(_prepare(src, rows, normalize))

    atomic_np_memmap_save(output_path, dtype=dtype, shape=shape, fill=_fill)
    _log_done("pca_per_model", shape, chunk_rows)
    return shape


def stream_pca_align(
    source_path: str | Path,
    output_path: str | Path,
    dim: int,
    train_items: np.ndarray,
    *,
    random_state: int | None = 42,
    chunk_rows: int = CHUNK_ROWS,
) -> tuple[int, int]:
    """Chunked counterpart of :func:`src.fusions.strategies.pca_align`.

    Reduces one native source to *dim* via a train-only PCA, writing the
    aligned ``.npy`` directly.  Mirrors ``pca_align`` exactly: no L2
    normalisation (the alignment operates on raw native features) and a
    ``float32`` output.  Used by the ``alignment.method: pca`` route,
    which otherwise loads every native matrix into the parent process at
    once.

    :param source_path: Native ``.npy`` to reduce.
    :param output_path: Destination for the aligned matrix.
    :param dim: Target dimensionality.
    :param train_items: Item indices forming the PCA fit set.
    :param random_state: Seed forwarded to scikit-learn.
    :param chunk_rows: Rows per transform pass.
    :returns: The shape of the array written.
    """
    source = np.load(source_path, mmap_mode="r")
    _validate_embeddings([source])
    n_rows = int(source.shape[0])
    fit_idx = np.asarray(train_items)

    fit_rows = np.asarray(source[fit_idx])
    k = min(dim, n_rows, int(source.shape[1]))
    k = min(k, *fit_rows.shape)
    label = f"pca_align[{Path(source_path).name}]"
    pca = fit_pca_on_rows(fit_rows, k, random_state, label, copy=False)
    del fit_rows

    shape = (n_rows, int(pca.n_components_))

    def _fill(out: np.memmap) -> None:
        for rows in _chunks(n_rows, chunk_rows):
            out[rows] = pca.transform(np.asarray(source[rows])).astype(np.float32)

    atomic_np_memmap_save(output_path, dtype=np.float32, shape=shape, fill=_fill)
    _log_done(label, shape, chunk_rows)
    return shape


def _assemble_fit_matrix(
    sources: list[np.ndarray],
    fit_idx: np.ndarray,
    normalize: bool,
    total_dim: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Build the ``(n_fit, total_dim)`` PCA fit matrix one source at a time.

    Equivalent to ``np.concatenate([...], axis=1)[fit_idx]`` but never
    holds a full-catalogue array: the peak is the fit matrix plus a
    single source's training rows.
    """
    fit_rows = np.empty((fit_idx.shape[0], total_dim), dtype=dtype)
    cursor = 0
    for src in sources:
        d = int(src.shape[1])
        fit_rows[:, cursor : cursor + d] = _prepare(src, fit_idx, normalize)
        cursor += d
    return fit_rows


def _log_done(strategy: str, shape: tuple[int, int], chunk_rows: int) -> None:
    logger.info(
        "  %s: streamed %s in chunks of %d rows",
        strategy,
        shape,
        chunk_rows,
    )
