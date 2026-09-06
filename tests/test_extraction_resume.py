"""Streaming extraction resumes from the durable row prefix (S03 / F11).

The progress sidecar recorded ``last_batch_index`` and a completed run
stored ``-1`` with ``rows_done == N``: a crash between that save and the
final rename made the next run resume at batch ZERO with ``row == N``.
Resume also trusted the batch index blindly — a different batch size
shifted every resumed row — and never checked that the part file was
produced by the same inputs and the same extraction recipe.

Every interruption here is a genuine exception raised at a durable
boundary; the final matrix is compared row by row with an uninterrupted
extraction of the same fake extractor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.extractors.base import BaseExtractor

DIM = 4


class _IdDataset(Dataset):
    """Images whose pixel value IS the item id; ``item_ids`` gives the order."""

    def __init__(self, ids: list[int]) -> None:
        self.item_ids = list(ids)

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int):
        item = self.item_ids[idx]
        return torch.full((3, 2, 2), float(item)), item


class _Model(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.projection = torch.nn.Identity()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(1, 2, 3)) + self.offset
        return self.projection(pooled[:, None].expand(-1, DIM))


class _Extractor(BaseExtractor):
    """Row of item ``i`` is ``i`` broadcast over ``DIM`` (weights ``fake/a``)."""

    weights_id = "fake/a"
    offset = 0.0

    def __init__(self) -> None:
        super().__init__(device="cpu")

    def _build_model(self):
        return _Model(self.offset)

    def _build_transform(self):
        from torchvision import transforms

        return transforms.ToTensor()


class _OtherWeights(_Extractor):
    """Same architecture, other weights: row of item ``i`` is ``i + 100``."""

    weights_id = "fake/b"
    offset = 100.0


def _loader(ids, batch_size: int) -> DataLoader:
    return DataLoader(_IdDataset(list(ids)), batch_size=batch_size, shuffle=False)


def _expected(ids, offset: float = 0.0) -> np.ndarray:
    col = np.asarray(ids, dtype=np.float32) + offset
    return np.broadcast_to(col[:, None], (len(col), DIM))


def _interrupt_after(ext: _Extractor, loader: DataLoader, n_batches: int):
    """Batch factory that dies when asked for batch ``n_batches``.

    Batch ``n_batches - 1`` has already been written to the memmap when
    the exception fires, so the interruption lands AFTER the last feature
    write and, unless ``save_every`` divides ``n_batches``, BEFORE the
    progress save covering it.
    """

    def factory(start: int):
        for batch in ext._iter_batches(loader, start, ext.model, ext._account_flops, "x"):
            if batch[0] >= n_batches:
                raise KeyboardInterrupt
            yield batch

    return factory


def _run_interrupted(ext, loader, base: Path, n_batches: int, save_every: int) -> None:
    with pytest.raises(KeyboardInterrupt):
        ext._extract_streaming(
            _interrupt_after(ext, loader, n_batches),
            loader,
            str(base),
            save_every=save_every,
            dtype=np.float32,
            empty_shape=(0, DIM),
        )


def _progress(base: Path) -> dict:
    return json.loads(Path(f"{base}.progress.json").read_text())


def _count_forwards(ext: _Extractor) -> list[int]:
    calls: list[int] = []
    original = ext.model.forward

    def spy(x):
        calls.append(int(x.shape[0]))
        return original(x)

    ext.model.forward = spy
    return calls


class TestCompleteButUnfinalised:
    def test_rows_done_n_finalises_instead_of_restarting_batch_zero(self, tmp_path: Path) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        loader = _loader(range(10), batch_size=4)
        # First run completes and the process dies BEFORE ``save`` renames.
        ext.extract_batch(loader, checkpoint_path=str(base), save_every=1)
        assert _progress(base)["rows_done"] == 10
        forwards = _count_forwards(ext)

        emb, ids = ext.extract_batch(loader, checkpoint_path=str(base), save_every=1)

        assert forwards == []  # nothing re-extracted
        assert ids == list(range(10))
        np.testing.assert_array_equal(np.asarray(emb), _expected(range(10)))
        ext.save(emb, ids, str(tmp_path / "out" / "ds_ext"))
        np.testing.assert_array_equal(
            np.load(tmp_path / "out" / "ds_ext.npy"), _expected(range(10))
        )

    def test_progress_is_persisted_only_for_flushed_rows(self, tmp_path: Path) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        loader = _loader(range(16), batch_size=4)

        _run_interrupted(ext, loader, base, n_batches=3, save_every=2)

        # Batches 0-1 were flushed and saved; batch 2 was written but its
        # progress save never happened: the durable prefix is 8 rows.
        progress = _progress(base)
        assert progress["rows_done"] == 8
        assert progress["item_ids"] == list(range(8))


class TestInterruptions:
    @pytest.mark.parametrize("n_batches", [1, 2, 3])
    def test_interrupt_after_feature_write_before_progress_save(
        self, tmp_path: Path, n_batches: int
    ) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        loader = _loader(range(15), batch_size=4)  # trailing partial batch

        _run_interrupted(ext, loader, base, n_batches=n_batches, save_every=2)
        emb, ids = ext.extract_batch(loader, checkpoint_path=str(base), save_every=2)

        assert ids == list(range(15))
        np.testing.assert_array_equal(np.asarray(emb), _expected(range(15)))

    def test_incomplete_last_block_is_rewritten_from_the_durable_boundary(
        self, tmp_path: Path
    ) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        loader = _loader(range(16), batch_size=4)
        _run_interrupted(ext, loader, base, n_batches=3, save_every=2)
        # Simulate a torn last block: only part of batch 2 reached disk.
        part = np.lib.format.open_memmap(f"{base}.part.npy", mode="r+")
        part[8:10] = -1.0
        part[10:] = 0.0
        part.flush()
        del part

        emb, ids = ext.extract_batch(loader, checkpoint_path=str(base), save_every=2)

        assert ids == list(range(16))
        np.testing.assert_array_equal(np.asarray(emb), _expected(range(16)))

    def test_interrupt_after_final_progress_save_then_save_finalises(self, tmp_path: Path) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        loader = _loader(range(9), batch_size=4)
        ext.extract_batch(loader, checkpoint_path=str(base), save_every=1)
        assert _progress(base)["last_batch_index"] == -1

        emb, ids = ext.extract_batch(loader, checkpoint_path=str(base), save_every=1)
        ext.save(emb, ids, str(tmp_path / "out" / "ds_ext"))

        np.testing.assert_array_equal(np.load(tmp_path / "out" / "ds_ext.npy"), _expected(range(9)))
        assert not Path(f"{base}.part.npy").exists()
        assert not Path(f"{base}.progress.json").exists()


class TestBatchSizeChange:
    @pytest.mark.parametrize("resume_batch_size", [2, 3, 8])
    def test_resume_with_another_batch_size_keeps_row_content(
        self, tmp_path: Path, resume_batch_size: int
    ) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        _run_interrupted(ext, _loader(range(14), batch_size=4), base, n_batches=3, save_every=1)
        assert _progress(base)["rows_done"] == 12

        emb, ids = ext.extract_batch(
            _loader(range(14), batch_size=resume_batch_size),
            checkpoint_path=str(base),
            save_every=1,
        )

        assert ids == list(range(14))
        np.testing.assert_array_equal(np.asarray(emb), _expected(range(14)))


class TestIdentity:
    def test_changed_input_order_restarts_clean(self, tmp_path: Path) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        _run_interrupted(ext, _loader(range(12), batch_size=4), base, n_batches=2, save_every=1)
        reordered = list(reversed(range(12)))  # same N, other order

        emb, ids = ext.extract_batch(
            _loader(reordered, batch_size=4), checkpoint_path=str(base), save_every=1
        )

        assert ids == reordered
        np.testing.assert_array_equal(np.asarray(emb), _expected(reordered))

    def test_changed_recipe_restarts_clean(self, tmp_path: Path) -> None:
        base = tmp_path / "ds_ext"
        loader = _loader(range(12), batch_size=4)
        _run_interrupted(_Extractor(), loader, base, n_batches=2, save_every=1)

        emb, ids = _OtherWeights().extract_batch(loader, checkpoint_path=str(base), save_every=1)

        assert ids == list(range(12))
        np.testing.assert_array_equal(np.asarray(emb), _expected(range(12), offset=100.0))

    def test_changed_dtype_restarts_clean(self, tmp_path: Path) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        loader = _loader(range(8), batch_size=4)
        _run_interrupted(ext, loader, base, n_batches=1, save_every=1)

        emb, ids = ext._extract_streaming(
            lambda start: ext._iter_batches(loader, start, ext.model, ext._account_flops, "x"),
            loader,
            str(base),
            save_every=1,
            dtype=np.float16,
            empty_shape=(0, DIM),
        )

        assert emb.dtype == np.float16
        assert ids == list(range(8))
        np.testing.assert_array_equal(np.asarray(emb, dtype=np.float32), _expected(range(8)))


class TestCorruptProgress:
    @pytest.mark.parametrize(
        "payload",
        [
            '{"schema_version": 2, "rows_done": 4, "n_to',  # truncated
            "",  # 0-byte
            '{"n_total": 12, "rows_done": 4}',  # missing keys
            '{"schema_version": 2, "n_total": 12, "rows_done": 4, "item_ids": [0, 1]}',
        ],
    )
    def test_corrupt_or_truncated_progress_is_handled_explicitly(
        self, tmp_path: Path, payload: str
    ) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        loader = _loader(range(12), batch_size=4)
        _run_interrupted(ext, loader, base, n_batches=2, save_every=1)
        Path(f"{base}.progress.json").write_text(payload)

        emb, ids = ext.extract_batch(loader, checkpoint_path=str(base), save_every=1)

        assert ids == list(range(12))
        np.testing.assert_array_equal(np.asarray(emb), _expected(range(12)))

    def test_legacy_progress_without_identity_is_not_trusted(self, tmp_path: Path) -> None:
        ext = _Extractor()
        base = tmp_path / "ds_ext"
        loader = _loader(range(12), batch_size=4)
        _run_interrupted(ext, loader, base, n_batches=2, save_every=1)
        # Pre-S03 sidecar: batch index, no identity; poison the rows it claims.
        Path(f"{base}.progress.json").write_text(
            json.dumps(
                {"last_batch_index": 1, "rows_done": 8, "n_total": 12, "item_ids": list(range(8))}
            )
        )
        part = np.lib.format.open_memmap(f"{base}.part.npy", mode="r+")
        part[:8] = -1.0
        part.flush()
        del part

        emb, ids = ext.extract_batch(loader, checkpoint_path=str(base), save_every=1)

        assert ids == list(range(12))
        np.testing.assert_array_equal(np.asarray(emb), _expected(range(12)))
