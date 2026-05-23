"""Tests for the hardened atomic write primitive.

Pins two production crash modes:

1. A process killed mid-serialisation must never leave a 0-byte or
   truncated final file (later read as ``EOFError`` / shape mismatch).
2. On some networked filesystems the just-written temp dirent is not
   durable/visible when the immediate rename runs, so ``os.replace``
   raises ``FileNotFound`` and an uncaught error kills the run.  The
   hardened primitive fsyncs the file (and best-effort the dir) before
   replacing and retries the replace on transient OS errors.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.utils.atomic_io import atomic_np_save, atomic_write


def test_atomic_write_roundtrips_and_removes_tmp(tmp_path: Path) -> None:
    dest = tmp_path / "x.bin"

    atomic_write(lambda p: Path(p).write_bytes(b"payload"), dest)

    assert dest.read_bytes() == b"payload"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_leaves_no_destination_when_write_fails(tmp_path: Path) -> None:
    dest = tmp_path / "x.bin"

    def _boom(p: str) -> None:
        Path(p).write_bytes(b"partial")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError, match="killed mid-write"):
        atomic_write(_boom, dest)

    assert not dest.exists()
    assert list(tmp_path.glob("*")) == []


def test_atomic_write_retries_replace_on_transient_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "x.bin"
    calls = {"n": 0}
    real_replace = os.replace

    def _flaky_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise FileNotFoundError(2, "No such file or directory")
        real_replace(src, dst)

    monkeypatch.setattr("src.utils.atomic_io.os.replace", _flaky_replace)

    atomic_write(lambda p: Path(p).write_bytes(b"ok"), dest, retries=5)

    assert dest.read_bytes() == b"ok"
    assert calls["n"] == 3


def test_atomic_write_raises_after_retries_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "x.bin"

    def _always_fail(src: str, dst: str) -> None:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr("src.utils.atomic_io.os.replace", _always_fail)

    with pytest.raises(FileNotFoundError):
        atomic_write(lambda p: Path(p).write_bytes(b"ok"), dest, retries=3)

    assert not dest.exists()
    assert list(tmp_path.glob("*")) == []


def test_atomic_np_save_still_roundtrips(tmp_path: Path) -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    dest = tmp_path / "emb.npy"

    atomic_np_save(array, dest)

    assert np.array_equal(np.load(dest), array)
