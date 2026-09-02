"""Pooled extraction streams to disk and survives a kill mid-run.

The accumulate-and-pickle checkpoint OOM-killed the container on
amazon_women x resnet50 (cgroup OOM at 2026-08-29 01:14) and left a
0-byte checkpoint that crashed the next resume with ``EOFError``.  With a
``checkpoint_path`` the pooled path now streams into an fp32
``<base>.part.npy`` memmap with a progress sidecar, exactly like the
component path, and ignores stale/corrupt legacy checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.test_component_streaming import DIM, _loader, _TraceableExtractor


def _expected_pooled(n: int) -> np.ndarray:
    return np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, DIM))


class TestPooledStreaming:
    def test_streams_to_fp32_part_file(self, tmp_path: Path) -> None:
        ext = _TraceableExtractor()
        base = tmp_path / "ds_ext"

        emb, ids = ext.extract_batch(_loader(10), checkpoint_path=str(base), save_every=1)

        assert isinstance(emb, np.memmap)
        assert emb.dtype == np.float32
        assert Path(f"{base}.part.npy").exists()
        np.testing.assert_array_equal(np.asarray(emb), _expected_pooled(10))
        assert ids == list(range(10))

    def test_resume_after_interruption_matches_full_run(self, tmp_path: Path) -> None:
        ext = _TraceableExtractor()
        base = tmp_path / "ds_ext"
        loader = _loader(12, batch_size=4)

        def _killed_after_two_batches(start: int):
            for b in ext._iter_batches(loader, start, ext.model, ext._account_flops, "x"):
                if b[0] >= 2:
                    raise KeyboardInterrupt  # simulate the OOM kill
                yield b

        with pytest.raises(KeyboardInterrupt):
            ext._extract_streaming(
                _killed_after_two_batches,
                loader,
                str(base),
                save_every=1,
                dtype=np.float32,
                empty_shape=(0, DIM),
            )
        assert Path(f"{base}.progress.json").exists()

        emb, ids = ext.extract_batch(loader, checkpoint_path=str(base), save_every=1)

        np.testing.assert_array_equal(np.asarray(emb), _expected_pooled(12))
        assert ids == list(range(12))

    def test_empty_legacy_checkpoint_is_discarded(self, tmp_path: Path) -> None:
        ext = _TraceableExtractor()
        base = tmp_path / "ds_ext"
        base.write_bytes(b"")  # what the OOM kill left behind

        emb, ids = ext.extract_batch(_loader(6), checkpoint_path=str(base))

        assert not base.exists()
        np.testing.assert_array_equal(np.asarray(emb), _expected_pooled(6))
        assert ids == list(range(6))

    def test_save_finalises_by_rename_and_clears_sidecar(self, tmp_path: Path) -> None:
        ext = _TraceableExtractor()
        base = tmp_path / "ckpt" / "ds_ext"
        emb, ids = ext.extract_batch(_loader(5), checkpoint_path=str(base))

        ext.save(emb, ids, str(tmp_path / "out" / "ds_ext"))

        assert not Path(f"{base}.part.npy").exists()
        assert not Path(f"{base}.progress.json").exists()
        saved = np.load(tmp_path / "out" / "ds_ext.npy")
        np.testing.assert_array_equal(saved, _expected_pooled(5))
