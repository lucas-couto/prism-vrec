"""``component_grid``: spatial pooling of ACF component maps at extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.extractors.base import BaseExtractor
from src.extractors.components import pool_components
from src.fusions import load_embedding
from src.utils.config_schema import validate_config

M_NATIVE = 16  # 4x4 region map


class _NativeDummyModel(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = torch.nn.Identity()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self._dim)


class _GridExtractor(BaseExtractor):
    supports_components = True
    extraction_point = "dummy"
    weights_id = "dummy"

    def __init__(self, native_dim: int = 3) -> None:
        self._dim = native_dim
        super().__init__(device="cpu")

    def _build_model(self):
        return _NativeDummyModel(self._dim)

    def _build_transform(self):
        from torchvision import transforms

        return transforms.ToTensor()

    def _forward_components(self, images: torch.Tensor) -> torch.Tensor:
        # region r of every item carries the value r (per channel), so the
        # pooled quadrants are checkable by hand.
        batch = images.shape[0]
        regions = torch.arange(M_NATIVE, dtype=torch.float32).view(1, M_NATIVE, 1)
        return regions.expand(batch, M_NATIVE, self._dim).clone()


class _Loader:
    """One-batch loader exposing ``dataset`` (streaming needs its length)."""

    def __init__(self, n: int) -> None:
        self.dataset = list(range(n))
        self._batch = (torch.zeros(n, 3, 8, 8), torch.arange(10, 10 + n))

    def __iter__(self):
        yield self._batch


def _fake_loader(n: int = 3) -> _Loader:
    return _Loader(n)


def test_pool_components_averages_each_quadrant_of_a_4x4_map() -> None:
    comp = torch.arange(16, dtype=torch.float32).view(1, 16, 1).expand(2, 16, 3).clone()

    pooled = pool_components(comp, grid=2)

    # 4x4 row-major map; quadrant means: TL {0,1,4,5}=2.5, TR {2,3,6,7}=4.5,
    # BL {8,9,12,13}=10.5, BR {10,11,14,15}=12.5.
    assert pooled.shape == (2, 4, 3)
    assert torch.allclose(pooled[0, :, 0], torch.tensor([2.5, 4.5, 10.5, 12.5]))
    assert torch.equal(pooled[0], pooled[1])


def test_pool_components_matches_torch_adaptive_pooling_on_a_7x7_map() -> None:
    comp = torch.randn(4, 49, 5)

    pooled = pool_components(comp, grid=3)

    expected = F.adaptive_avg_pool2d(comp.transpose(1, 2).reshape(4, 5, 7, 7), 3)
    expected = expected.reshape(4, 5, 9).transpose(1, 2)
    assert torch.allclose(pooled, expected)


def test_pool_components_keeps_input_for_none_or_native_grid() -> None:
    comp = torch.randn(2, 49, 4)

    assert pool_components(comp, None) is comp
    assert pool_components(comp, 7) is comp


@pytest.mark.parametrize("bad", [(torch.randn(2, 12, 4), 2), (torch.randn(2, 49, 4), 8)])
def test_pool_components_rejects_non_square_maps_and_upsampling(bad) -> None:
    comp, grid = bad

    with pytest.raises(ValueError):
        pool_components(comp, grid)


def test_extractor_pools_components_when_component_grid_is_set() -> None:
    extractor = _GridExtractor()
    extractor.component_grid = 2

    components, ids = extractor.extract_components_batch(_fake_loader())

    assert components.shape == (3, 4, 3)
    assert components.dtype == np.float16
    assert ids == [10, 11, 12]
    assert np.allclose(components[0, :, 0], [2.5, 4.5, 10.5, 12.5])


def test_extractor_keeps_native_regions_without_component_grid() -> None:
    extractor = _GridExtractor()

    components, _ = extractor.extract_components_batch(_fake_loader())

    assert components.shape == (3, M_NATIVE, 3)


def test_streaming_extraction_pools_too(tmp_path: Path) -> None:
    extractor = _GridExtractor()
    extractor.component_grid = 2

    components, ids = extractor.extract_components_batch(
        _fake_loader(), checkpoint_path=str(tmp_path / "x_comp"), save_every=1
    )

    assert components.shape == (3, 4, 3)
    assert np.allclose(np.asarray(components[1, :, 1]), [2.5, 4.5, 10.5, 12.5])


def test_load_embedding_memmaps_component_artifacts(tmp_path: Path) -> None:
    arr = np.random.default_rng(0).standard_normal((6, 4, 3)).astype(np.float16)
    path = tmp_path / "resnet50_comp.npy"
    np.save(path, arr)
    (tmp_path / "resnet50.npy").write_bytes(b"")  # unrelated sibling

    loaded = load_embedding(path)

    assert isinstance(loaded, np.memmap)
    assert loaded.dtype == np.float16
    assert np.array_equal(np.asarray(loaded), arr)


def test_schema_accepts_component_grid_and_rejects_zero() -> None:
    base = {"datasets": [], "component_grid": 2}

    assert validate_config(base)["component_grid"] == 2
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        validate_config({"datasets": [], "component_grid": 0})


def test_meta_records_the_grid(tmp_path: Path) -> None:
    from src.steps.extract import _write_meta

    extractor = _GridExtractor()
    extractor.component_grid = 2
    _write_meta(
        extractor,
        "dummy",
        tmp_path / "dummy_comp.npy",
        {"kind": "components", "n_components": 4, "component_grid": 2, "pooling": "adaptive_avg"},
    )

    meta = json.loads((tmp_path / "dummy_comp.meta.json").read_text())
    assert meta["component_grid"] == 2 and meta["n_components"] == 4
