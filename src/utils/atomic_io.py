"""Atomic, durable on-disk persistence helpers.

Two crash modes this module guards against on distributed filesystems:

1. A process killed mid-serialisation leaving a 0-byte / truncated
   final file (later read as ``EOFError`` or a shape mismatch).
2. ``tmp.rename(final)`` without fsync: on some networked filesystems
   the freshly written temp dirent is not durable/visible by the time
   the immediate rename runs, so ``os.replace`` raises
   ``FileNotFoundError`` and kills the run.

:func:`atomic_write` serialises into a PID-scoped sibling temp file,
fsyncs the file (and best-effort the directory) so the bytes and the
dirent are committed before the rename, then ``os.replace``-s with a
bounded retry for residual metadata-propagation lag.  The destination
is always either the previous file or the complete new one.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def _fsync_path(path: Path) -> None:
    """fsync *path* (a file). Best-effort: never raises."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _fsync_dir(path: Path) -> None:
    """fsync a directory so a new dirent is durable. Best-effort."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some filesystems / FUSE mounts reject directory fsync.
        pass


def atomic_write(
    write_fn: Callable[[str], None],
    path: str | Path,
    retries: int = 6,
) -> None:
    """Write *path* atomically and durably.

    ``write_fn`` receives the temp-file path and must write the full
    contents to it. The temp file is fsynced, the parent directory is
    fsynced best-effort, then it is ``os.replace``-d onto *path* with a
    bounded retry on transient OS errors (networked-FS dirent propagation).

    Args:
        write_fn: Callback that writes the payload to the given path.
        path: Final destination; its parent is created if absent.
        retries: Max ``os.replace`` attempts before giving up.

    Raises:
        Exception: Re-raises the original error after removing the temp
            file, so *path* is never left partial.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")

    try:
        write_fn(str(tmp))
        _fsync_path(tmp)
        _fsync_dir(dest.parent)

        last: OSError | None = None
        for attempt in range(retries):
            try:
                os.replace(tmp, dest)
                return
            except OSError as exc:
                last = exc
                time.sleep(0.25 * (attempt + 1))
        raise last  # type: ignore[misc]
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def atomic_np_save(array: np.ndarray, path: str | Path) -> None:
    """Persist *array* to *path* in ``.npy`` format atomically.

    Bytes written are identical to ``numpy.save(path, array)``; only the
    write is made crash-safe via :func:`atomic_write`.

    Args:
        array: Array to serialise.
        path: Destination file; its parent directory is created if absent.
    """

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as handle:
            np.save(handle, array)

    atomic_write(_write, path)


def atomic_np_memmap_save(
    path: str | Path,
    *,
    dtype: Any,
    shape: tuple[int, ...],
    fill: Callable[[np.memmap], None],
) -> None:
    """Build a ``.npy`` on disk through a writable memmap, atomically.

    The streaming counterpart of :func:`atomic_np_save`: instead of
    taking a fully materialised array, it allocates the destination on
    disk and hands *fill* a writable view to populate in pieces.  Peak
    memory is therefore whatever *fill* holds at once, not the size of
    the result — the difference between a few hundred MB and tens of GB
    for a catalogue-sized fusion.

    The result is an ordinary ``.npy`` that :func:`numpy.load` reads
    back as the equivalent array, and the same crash-safety guarantees
    apply: the destination is either the previous file or the complete
    new one.

    Args:
        path: Destination file; its parent directory is created if absent.
        dtype: dtype of the array to create.
        shape: Shape of the array to create.
        fill: Callback receiving the writable memmap.  It must populate
            every element; unwritten regions are zeros.

    Raises:
        Exception: Re-raises whatever *fill* raises, after removing the
            temp file, so *path* is never left partial.
    """

    def _write(tmp: str) -> None:
        handle = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape)
        try:
            fill(handle)
            handle.flush()
        finally:
            # Drop the mapping before the rename: on some platforms a
            # live mapping keeps the temp inode pinned.
            del handle

    atomic_write(_write, path)
