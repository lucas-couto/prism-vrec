"""Bounded, read-only access to per-item feature rows (SDD C01, task M01).

The recommenders historically received the whole feature matrix as an
in-memory array and registered it as a module buffer, so ``model.to(
device)`` moved every catalogue row to the GPU and a fusion sidecar was
materialised by stacking or concatenating all of its sources first.
This module is the internal adapter that lets the same models read only
the rows a forward pass needs:

* :class:`NpyFeatureSource` — a ``.npy`` file opened as a read-only
  memory map; rows are gathered on demand and the on-disk dtype is kept
  (fp16 component artifacts stay fp16 until the consumer casts them).
* :class:`ConcatFeatureSource` — ordered native sources of differing
  widths, concatenated along the feature axis **per gathered row** (the
  learned-alignment sidecar).  Carries the fusion recipe the recommender
  needs, mirroring :class:`src.fusions.online.RaggedSources`.
* :class:`StackedFeatureSource` — ordered equal-shape sources stacked
  along a new axis 1 **per gathered row** (the non-learned sidecar).
* :class:`ArrayFeatureSource` — an in-memory array behind the same
  interface; the dense reference for tests and for callers that already
  hold the matrix.

Every source is immutable and host-owned: :meth:`FeatureSource.read_rows`
returns a fresh contiguous array the caller may move to any device.
Memory maps are opened lazily in the process that reads, never inherited
through pickling (``spawn`` workers reopen their own handle), and
released by :meth:`FeatureSource.close`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FeatureSource(Protocol):
    """Read-only, row-addressable feature store (C01)."""

    @property
    def shape(self) -> tuple[int, ...]:
        """``(n_items, *trailing)`` of the full source."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """Element dtype of the rows :meth:`read_rows` returns."""
        ...

    def read_rows(self, item_ids: np.ndarray) -> np.ndarray:
        """Gather ``item_ids`` (1-D integers) in caller order.

        Returns a host-owned array of shape ``(len(item_ids), *trailing)``;
        an empty request keeps the trailing shape and the dtype.  Out of
        range or non-integer ids raise.
        """
        ...

    def close(self) -> None:
        """Release any file handle; the source stays usable (reopens)."""
        ...


def is_feature_source(obj: Any) -> bool:
    """Whether *obj* is a lazy source rather than an array-like."""
    return isinstance(obj, FeatureSource) and not isinstance(obj, np.ndarray)


def source_bytes(source: Any) -> int:
    """Raw payload bytes of a source or array: ``N * prod(trailing) * itemsize``.

    A sidecar's own file size says nothing about its sources; this is
    the number the memory ledger (SPEC §"Memory ledger") needs.
    """
    shape = tuple(int(s) for s in source.shape)
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(source.dtype).itemsize)


def _validate_ids(item_ids: Any, n_rows: int) -> np.ndarray:
    """Return *item_ids* as a 1-D int64 array, failing on invalid indices."""
    ids = np.asarray(item_ids)
    if ids.ndim != 1:
        raise ValueError(f"item_ids must be 1-D, got shape {ids.shape}.")
    if ids.size and not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"item_ids must be integers, got dtype {ids.dtype}.")
    ids = ids.astype(np.int64, copy=False)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= n_rows):
        raise IndexError(
            f"item_ids out of range for a source with {n_rows} rows "
            f"(min={int(ids.min())}, max={int(ids.max())})."
        )
    return ids


class ArrayFeatureSource:
    """In-memory array behind the :class:`FeatureSource` interface."""

    def __init__(self, array: np.ndarray) -> None:
        arr = np.asarray(array)
        if arr.ndim < 2:
            raise ValueError(f"feature arrays must be at least 2-D, got shape {arr.shape}.")
        self._array = arr

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    def read_rows(self, item_ids: np.ndarray) -> np.ndarray:
        ids = _validate_ids(item_ids, self.shape[0])
        return np.ascontiguousarray(self._array[ids])

    def close(self) -> None:
        return None


class NpyFeatureSource:
    """Read-only memory map over one ``.npy`` file, opened per process.

    The header is read once at construction (shape/dtype validation and
    a clear error for a missing file); the map itself is opened on the
    first read in whichever process performs it and dropped from the
    pickled state, so a ``spawn``-ed worker never inherits a parent's
    handle.  A ``fork``-ed child that reads reopens as well (the owner
    pid is recorded).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._shape, self._dtype = _read_npy_header(self._path)
        if len(self._shape) < 2:
            raise ValueError(
                f"{self._path}: feature files must be at least 2-D, got shape {self._shape}."
            )
        self._mmap: np.memmap | None = None
        self._owner_pid: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def is_open(self) -> bool:
        return self._mmap is not None and self._owner_pid == os.getpid()

    def _map(self) -> np.memmap:
        if not self.is_open:
            self._mmap = np.load(self._path, mmap_mode="r")
            self._owner_pid = os.getpid()
        assert self._mmap is not None
        return self._mmap

    def read_rows(self, item_ids: np.ndarray) -> np.ndarray:
        ids = _validate_ids(item_ids, self._shape[0])
        # Fancy indexing on a memmap copies exactly the requested rows
        # into a plain, host-owned ndarray; nothing else is paged in.
        return np.ascontiguousarray(self._map()[ids])

    def close(self) -> None:
        self._mmap = None
        self._owner_pid = None

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_mmap"] = None
        state["_owner_pid"] = None
        return state

    def __repr__(self) -> str:
        return f"NpyFeatureSource({str(self._path)!r}, shape={self._shape}, dtype={self._dtype})"


def _read_npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    """``(shape, dtype)`` from the ``.npy`` header without mapping the data."""
    if not path.exists():
        raise FileNotFoundError(f"feature file not found: {path}")
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran_order, dtype = reader(handle)
    if fortran_order:
        raise ValueError(f"{path}: Fortran-ordered feature files are not supported.")
    return tuple(int(s) for s in shape), np.dtype(dtype)


class _MultiSource:
    """Shared plumbing for sidecars: ordered sources, common row count."""

    def __init__(self, sources: list[FeatureSource]) -> None:
        if not sources:
            raise ValueError("a multi-source feature needs at least one source.")
        n_rows = {int(s.shape[0]) for s in sources}
        if len(n_rows) != 1:
            raise ValueError(f"sources disagree on n_items ({sorted(n_rows)}).")
        self._sources = list(sources)
        self._dtype = np.result_type(*(s.dtype for s in sources))

    @property
    def sources(self) -> tuple[FeatureSource, ...]:
        return tuple(self._sources)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def close(self) -> None:
        for source in self._sources:
            source.close()


class ConcatFeatureSource(_MultiSource):
    """Ordered native sources concatenated along the feature axis per row.

    Carries the learned-alignment recipe (``source_dims``, ``strategy``,
    ``aligned_dim``, ``normalize``, ``fusion_kwargs``) so the recommender
    can build its :class:`~src.fusions.online.LearnedAlignmentFusion`
    from the same attributes it reads off a
    :class:`~src.fusions.online.RaggedSources` array.
    """

    def __init__(
        self,
        sources: list[FeatureSource],
        *,
        strategy: str,
        aligned_dim: int,
        normalize: bool = True,
        fusion_kwargs: dict | None = None,
    ) -> None:
        super().__init__(sources)
        for source in self._sources:
            if len(source.shape) != 2:
                raise ValueError(
                    f"learned-alignment sources must be 2-D, got shape {source.shape}."
                )
        self.source_dims = [int(s.shape[1]) for s in self._sources]
        self.strategy = str(strategy)
        self.aligned_dim = int(aligned_dim)
        self.normalize = bool(normalize)
        self.fusion_kwargs = dict(fusion_kwargs or {})

    @property
    def shape(self) -> tuple[int, ...]:
        return (int(self._sources[0].shape[0]), int(sum(self.source_dims)))

    def read_rows(self, item_ids: np.ndarray) -> np.ndarray:
        ids = _validate_ids(item_ids, self.shape[0])
        parts = [s.read_rows(ids).astype(self._dtype, copy=False) for s in self._sources]
        return np.concatenate(parts, axis=1)


class StackedFeatureSource(_MultiSource):
    """Ordered equal-shape sources stacked along a new axis 1 per row.

    ``read_rows`` returns ``(B, M, *trailing)`` exactly as
    ``np.stack(all_sources, axis=1)[ids]`` would, without ever building
    the full stack.  With ``normalize=True`` every gathered row of every
    source is L2-normalised first (SDD S02: the non-learned online
    sidecar's pre-fusion normalisation, mirroring the eager
    :class:`src.fusions.online.StackedSources`), with the offline
    :func:`~src.fusions.strategies.l2_normalize` rule for zero rows.
    ``recipe_version`` / ``sidecar_recipe_version`` carry the same recipe
    identity as the eager array.
    """

    def __init__(
        self,
        sources: list[FeatureSource],
        *,
        normalize: bool = False,
        sidecar_recipe_version: int | None = None,
    ) -> None:
        super().__init__(sources)
        first = tuple(self._sources[0].shape)
        for source in self._sources[1:]:
            if tuple(source.shape) != first:
                raise ValueError(
                    f"stacked sources must share a shape: {tuple(source.shape)} != {first}."
                )
        if normalize and len(first) != 2:
            raise ValueError(
                f"per-source normalisation needs 2-D sources, got shape {first} "
                "(component stacks are consumed raw)."
            )
        self.normalize = bool(normalize)
        self.sidecar_recipe_version = (
            None if sidecar_recipe_version is None else int(sidecar_recipe_version)
        )

    @property
    def recipe_version(self) -> int:
        from src.fusions.online import SIDECAR_RECIPE_VERSION  # avoid cycle

        return SIDECAR_RECIPE_VERSION

    @property
    def shape(self) -> tuple[int, ...]:
        first = self._sources[0].shape
        return (int(first[0]), len(self._sources), *tuple(int(s) for s in first[1:]))

    def read_rows(self, item_ids: np.ndarray) -> np.ndarray:
        ids = _validate_ids(item_ids, self.shape[0])
        parts = [s.read_rows(ids).astype(self._dtype, copy=False) for s in self._sources]
        if self.normalize:
            from src.fusions.strategies import l2_normalize  # avoid cycle

            parts = [l2_normalize(part) for part in parts]
        return np.stack(parts, axis=1)


__all__ = [
    "ArrayFeatureSource",
    "ConcatFeatureSource",
    "FeatureSource",
    "NpyFeatureSource",
    "StackedFeatureSource",
    "is_feature_source",
    "source_bytes",
]
