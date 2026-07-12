"""Tests for the BaseExtractor component-feature capability.

The batch/save logic is exercised with a lightweight dummy extractor (no
backbone download); a separate import-guarded test checks ResNet-50's
real ``M=49`` spatial grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor

M_COMPONENTS = 5


class _NativeDummyModel(nn.Module):
    """Backbone stand-in emitting zeros at a fixed native dim."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.zeros(x.shape[0], self.dim))


class _DummyComponentExtractor(BaseExtractor):
    """Minimal extractor exposing deterministic component features."""

    supports_components = True

    def __init__(self, native_dim: int = 4) -> None:
        self._dim = native_dim
        super().__init__(device="cpu")

    def _build_model(self):
        return _NativeDummyModel(self._dim)

    def _build_transform(self):
        from torchvision import transforms

        return transforms.ToTensor()

    def _forward_components(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        return torch.zeros(batch, M_COMPONENTS, self.native_dim)


class _PooledOnlyExtractor(BaseExtractor):
    """Extractor that does not advertise component support."""

    def __init__(self) -> None:
        self._dim = 4
        super().__init__(device="cpu")

    def _build_model(self):
        return _NativeDummyModel(self._dim)

    def _build_transform(self):
        from torchvision import transforms

        return transforms.ToTensor()


def _fake_loader() -> list:
    return [
        (torch.zeros(2, 3, 8, 8), [10, 11]),
        (torch.zeros(1, 3, 8, 8), [12]),
    ]


def test_base_extractor_defaults_to_no_components() -> None:
    assert BaseExtractor.supports_components is False


def test_unsupported_extractor_raises_on_components() -> None:
    extractor = _PooledOnlyExtractor()

    with pytest.raises(NotImplementedError):
        extractor._forward_components(torch.zeros(1, 3, 8, 8))


def test_extract_components_batch_stacks_n_m_d() -> None:
    extractor = _DummyComponentExtractor(native_dim=4)

    components, item_ids = extractor.extract_components_batch(_fake_loader())

    assert components.shape == (3, M_COMPONENTS, 4)
    assert item_ids == [10, 11, 12]


def test_save_components_writes_3d_npy_and_ids(tmp_path: Path) -> None:
    extractor = _DummyComponentExtractor(native_dim=4)
    components, item_ids = extractor.extract_components_batch(_fake_loader())

    out = tmp_path / "resnet50_D4_comp"
    extractor.save_components(components, item_ids, str(out))

    saved = np.load(out.with_suffix(".npy"))
    ids = json.loads((tmp_path / "resnet50_D4_comp_ids.json").read_text())
    assert saved.shape == (3, M_COMPONENTS, 4)
    assert ids == [10, 11, 12]


@pytest.mark.slow
def test_resnet50_exposes_49_components() -> None:
    pytest.importorskip("torchvision")
    from src.extractors.resnet import ResNet50Extractor

    extractor = ResNet50Extractor(device="cpu", output_dim=16)
    assert extractor.supports_components is True

    with torch.no_grad():
        components = extractor._forward_components(torch.randn(2, 3, 224, 224))

    assert components.shape == (2, 49, 16)
