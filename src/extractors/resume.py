"""Resume point of a streaming extraction: the durable row prefix plus identity.

A streaming extraction writes batches into ``<base>.part.npy`` and
records progress in ``<base>.progress.json``.  Resuming is only sound
when three things hold (SDD S03 / F11, requirements Q13–Q14):

* ``rows_done`` is treated as the length of the *durable* row prefix
  (rows flushed before the progress save), never as a batch index;
* the part file was produced from the same ordered inputs, by the same
  extraction recipe (backbone, weights, transform, precision, component
  grid) and with the same on-disk dtype; and
* a batch-size change maps the prefix to a row offset (rewinding to the
  last batch boundary) instead of trusting a stale batch index.

A completed-but-unfinalised part file (``rows_done == n_total``) is
finalised, not re-extracted.  Anything that cannot be validated —
corrupt or truncated JSON, a legacy sidecar without identity, a
mismatching digest — restarts from row zero with an explicit warning:
the part file is a working file, and recomputation is always safe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.item_order import item_order_digest
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Version of the ``.progress.json`` sidecar written by this module.
#: Version 1 (implicit, no ``schema_version`` key) carried a batch index
#: and no identity; it is recognised and never trusted.
PROGRESS_SCHEMA_VERSION = 2

_REQUIRED_KEYS = (
    "last_batch_index",
    "rows_done",
    "n_total",
    "item_ids",
    "batch_size",
    "input_digest",
    "recipe_digest",
    "dtype",
    "shape",
    "complete",
)


class ProgressError(ValueError):
    """The progress sidecar is corrupt, truncated, legacy or inconsistent."""


@dataclass(frozen=True)
class StreamIdentity:
    """What a part file must have been produced from to be resumable."""

    n_total: int
    dtype: str
    input_digest: str | None
    recipe_digest: str
    batch_size: int | None


@dataclass
class ResumePoint:
    """Where :meth:`BaseExtractor._extract_streaming` continues from."""

    memmap: np.memmap | None
    start_batch: int
    row: int
    item_ids: list
    complete: bool


def _dataset_input_digest(dataloader, n_total: int) -> str | None:
    """Digest of the dataset's ordered ``item_ids`` when it exposes them."""
    ids = getattr(getattr(dataloader, "dataset", None), "item_ids", None)
    if ids is None or len(ids) != n_total:
        return None
    return item_order_digest(list(ids))


def recipe_digest(metadata: dict, *, dtype: str, kind: str, component_grid: int | None) -> str:
    """SHA-256 over the canonical JSON of the extraction recipe."""
    recipe = {
        "kind": kind,
        "dtype": dtype,
        "component_grid": component_grid,
        **{key: metadata.get(key) for key in sorted(metadata)},
    }
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stream_identity(
    dataloader,
    metadata: dict,
    *,
    n_total: int,
    dtype,
    kind: str,
    component_grid: int | None,
) -> StreamIdentity:
    """Build the identity of the stream about to be written (or resumed)."""
    dtype_name = str(np.dtype(dtype))
    return StreamIdentity(
        n_total=n_total,
        dtype=dtype_name,
        input_digest=_dataset_input_digest(dataloader, n_total),
        recipe_digest=recipe_digest(
            metadata, dtype=dtype_name, kind=kind, component_grid=component_grid
        ),
        batch_size=getattr(dataloader, "batch_size", None),
    )


def progress_payload(
    identity: StreamIdentity,
    *,
    last_batch_index: int,
    rows_done: int,
    item_ids: list,
    shape: tuple,
    complete: bool,
) -> dict:
    """The v2 progress sidecar; ``rows_done`` must count flushed rows only."""
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "last_batch_index": last_batch_index,
        "rows_done": rows_done,
        "n_total": identity.n_total,
        "item_ids": list(item_ids),
        "batch_size": identity.batch_size,
        "input_digest": identity.input_digest,
        "recipe_digest": identity.recipe_digest,
        "dtype": identity.dtype,
        "shape": [int(s) for s in shape],
        "complete": complete,
    }


def _load_progress(progress_path: Path) -> dict:
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProgressError(f"unreadable or truncated JSON ({exc})") from exc
    if not isinstance(progress, dict):
        raise ProgressError("payload is not a JSON object")
    version = progress.get("schema_version")
    if version != PROGRESS_SCHEMA_VERSION:
        raise ProgressError(
            f"schema version {version!r} carries no verifiable identity "
            f"(expected {PROGRESS_SCHEMA_VERSION})"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in progress]
    if missing:
        raise ProgressError(f"missing keys {missing}")
    rows_done, n_total = progress["rows_done"], progress["n_total"]
    if not (isinstance(rows_done, int) and isinstance(n_total, int) and 0 <= rows_done <= n_total):
        raise ProgressError(f"rows_done={rows_done!r} outside [0, n_total={n_total!r}]")
    if not isinstance(progress["item_ids"], list) or len(progress["item_ids"]) != rows_done:
        raise ProgressError("item_ids length does not equal rows_done")
    return progress


def _identity_mismatch(progress: dict, identity: StreamIdentity) -> str | None:
    if progress["n_total"] != identity.n_total:
        return f"catalogue size changed ({progress['n_total']} -> {identity.n_total})"
    if progress["dtype"] != identity.dtype:
        return f"on-disk dtype changed ({progress['dtype']} -> {identity.dtype})"
    if identity.input_digest is None or progress["input_digest"] is None:
        return "ordered-input identity unknown (dataset exposes no item_ids)"
    if progress["input_digest"] != identity.input_digest:
        return "ordered inputs changed (same size, different item order or ids)"
    if progress["recipe_digest"] != identity.recipe_digest:
        return "extraction recipe changed (weights, transform, precision or component grid)"
    return None


def _restart(part_path: Path, progress_path: Path, reason: str) -> ResumePoint:
    logger.warning(
        "Extraction checkpoint %s is not resumable (%s); restarting from row 0.",
        part_path.name,
        reason,
    )
    part_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return ResumePoint(None, 0, 0, [], False)


def _open_part(part_path: Path, progress: dict, identity: StreamIdentity) -> np.memmap | None:
    """Reopen the part file; ``None`` when it disagrees with the sidecar."""
    try:
        candidate = np.lib.format.open_memmap(part_path, mode="r+")
    except (ValueError, OSError):
        return None
    if (
        list(candidate.shape) != list(progress["shape"])
        or candidate.shape[0] != identity.n_total
        or str(candidate.dtype) != identity.dtype
    ):
        del candidate
        return None
    return candidate


def resume_state(part_path: Path, progress_path: Path, identity: StreamIdentity) -> ResumePoint:
    """Validate the part file + sidecar and return where to continue.

    :returns: A :class:`ResumePoint`; ``complete`` is set when every row
        is already on disk and only the finalisation is pending.
    """
    if not (part_path.exists() and progress_path.exists()):
        if part_path.exists() or progress_path.exists():
            return _restart(part_path, progress_path, "part file and progress sidecar do not pair")
        return ResumePoint(None, 0, 0, [], False)
    try:
        progress = _load_progress(progress_path)
    except ProgressError as exc:
        return _restart(part_path, progress_path, f"corrupt progress sidecar: {exc}")
    mismatch = _identity_mismatch(progress, identity)
    if mismatch:
        return _restart(part_path, progress_path, mismatch)
    memmap = _open_part(part_path, progress, identity)
    if memmap is None:
        return _restart(part_path, progress_path, "part file shape/dtype disagree with the sidecar")

    rows_done: int = progress["rows_done"]
    item_ids = list(progress["item_ids"])
    if rows_done == identity.n_total:
        logger.info("  resume: %s is complete; finalising without re-extraction", part_path.name)
        return ResumePoint(memmap, 0, rows_done, item_ids, True)
    if identity.batch_size is None:
        del memmap
        return _restart(
            part_path,
            progress_path,
            "dataloader has no fixed batch_size; the durable row prefix cannot be "
            "mapped to a batch offset",
        )
    resume_row = (rows_done // identity.batch_size) * identity.batch_size
    if resume_row < rows_done:
        logger.info(
            "  resume: rewinding %d -> %d rows to the last batch boundary (batch_size=%d)",
            rows_done,
            resume_row,
            identity.batch_size,
        )
    logger.info(
        "  resume: %d/%d rows already on disk (%s)", resume_row, identity.n_total, part_path.name
    )
    return ResumePoint(
        memmap, resume_row // identity.batch_size, resume_row, item_ids[:resume_row], False
    )


__all__ = [
    "PROGRESS_SCHEMA_VERSION",
    "ProgressError",
    "ResumePoint",
    "StreamIdentity",
    "progress_payload",
    "recipe_digest",
    "resume_state",
    "stream_identity",
]
