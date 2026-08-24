"""Streaming component extraction: bounded RAM, fp16 on disk, resume.

The in-RAM accumulation OOM-killed the container on real catalogues
(resnet50 x amazon_fashion = 67 GB fp32 against a 24 GB cgroup) and its
checkpoint re-serialised the whole accumulated matrix.  With a
``checkpoint_path`` the extraction now streams every batch into an fp16
``<base>.part.npy`` memmap with a JSON progress sidecar, and
``save_components`` finalises by atomic rename.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.extractors.base import BaseExtractor

M = 3
DIM = 4


class _IdDataset(Dataset):
    """Images whose pixel value IS the item id, so outputs are traceable."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return torch.full((3, 2, 2), float(idx)), idx


class _TraceableModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x.mean(dim=(1, 2, 3))[:, None].expand(-1, DIM))


class _TraceableExtractor(BaseExtractor):
    """Component m of item i is ``i + m`` — every cell is checkable."""

    supports_components = True

    def __init__(self) -> None:
        super().__init__(device="cpu")

    def _build_model(self):
        return _TraceableModel()

    def _build_transform(self):
        from torchvision import transforms

        return transforms.ToTensor()

    def _forward_components(self, images: torch.Tensor) -> torch.Tensor:
        ids = images.mean(dim=(1, 2, 3))  # recover the item id
        offsets = torch.arange(M, dtype=torch.float32)
        return (ids[:, None] + offsets[None, :])[:, :, None].expand(-1, -1, DIM)


def _expected(n: int) -> np.ndarray:
    ids = np.arange(n, dtype=np.float32)
    offs = np.arange(M, dtype=np.float32)
    return np.broadcast_to((ids[:, None] + offs[None, :])[:, :, None], (n, M, DIM))


def _loader(n: int, batch_size: int = 4) -> DataLoader:
    return DataLoader(_IdDataset(n), batch_size=batch_size, shuffle=False)


class TestStreamingExtraction:
    def test_streams_to_fp16_part_file(self, tmp_path: Path) -> None:
        base = tmp_path / "resnet50_comp"

        components, ids = _TraceableExtractor().extract_components_batch(
            _loader(10), checkpoint_path=str(base), save_every=1
        )

        assert isinstance(components, np.memmap)
        assert components.dtype == np.float16
        assert components.shape == (10, M, DIM)
        assert ids == list(range(10))
        np.testing.assert_allclose(np.asarray(components, dtype=np.float32), _expected(10))
        assert (tmp_path / "resnet50_comp.part.npy").exists()

    def test_save_components_finalises_by_rename(self, tmp_path: Path) -> None:
        base = tmp_path / "resnet50_comp"
        extractor = _TraceableExtractor()
        components, ids = extractor.extract_components_batch(
            _loader(6), checkpoint_path=str(base), save_every=2
        )

        extractor.save_components(components, ids, str(base))

        final = np.load(base.with_suffix(".npy"))
        assert final.dtype == np.float16
        np.testing.assert_allclose(final.astype(np.float32), _expected(6))
        assert json.loads((tmp_path / "resnet50_comp_ids.json").read_text()) == list(range(6))
        # Working files are gone: renamed + progress cleaned up.
        assert not (tmp_path / "resnet50_comp.part.npy").exists()
        assert not (tmp_path / "resnet50_comp.progress.json").exists()

    def test_resume_skips_completed_batches_and_matches_full_run(self, tmp_path: Path) -> None:
        base = tmp_path / "resnet50_comp"
        extractor = _TraceableExtractor()

        # First attempt "dies" after 2 of 3 batches: simulate by rolling
        # the progress sidecar back to batch index 1 and poisoning the
        # rows a resume must NOT rewrite.
        extractor.extract_components_batch(_loader(12), checkpoint_path=str(base), save_every=1)
        progress_path = tmp_path / "resnet50_comp.progress.json"
        progress = json.loads(progress_path.read_text())
        progress.update(last_batch_index=1, rows_done=8, item_ids=list(range(8)))
        progress_path.write_text(json.dumps(progress))
        part = np.lib.format.open_memmap(tmp_path / "resnet50_comp.part.npy", mode="r+")
        part[8:] = -1.0  # rows the resume is responsible for
        part.flush()
        del part

        components, ids = extractor.extract_components_batch(
            _loader(12), checkpoint_path=str(base), save_every=1
        )

        assert ids == list(range(12))
        np.testing.assert_allclose(np.asarray(components, dtype=np.float32), _expected(12))

    def test_changed_catalogue_restarts_clean(self, tmp_path: Path) -> None:
        base = tmp_path / "resnet50_comp"
        extractor = _TraceableExtractor()
        extractor.extract_components_batch(_loader(8), checkpoint_path=str(base), save_every=1)

        components, ids = extractor.extract_components_batch(
            _loader(12), checkpoint_path=str(base), save_every=1
        )

        assert components.shape == (12, M, DIM)
        assert ids == list(range(12))

    def test_in_memory_path_without_checkpoint_still_works(self) -> None:
        components, ids = _TraceableExtractor().extract_components_batch(_loader(5))

        assert components.dtype == np.float16
        assert components.shape == (5, M, DIM)
        assert ids == list(range(5))
