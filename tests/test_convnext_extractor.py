"""Unit tests for ConvNeXt-Base extractor.

Mirrors the contract guarded by `tests/test_extractor_unfreeze.py` and
the shape contract documented in `BaseExtractor`.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def extractor():
    pytest.importorskip("timm")
    from src.extractors.convnext import ConvNeXtExtractor

    return ConvNeXtExtractor(device="cpu", output_dim=64)


def test_convnext_is_a_base_extractor():
    from src.extractors.base import BaseExtractor
    from src.extractors.convnext import ConvNeXtExtractor

    assert issubclass(ConvNeXtExtractor, BaseExtractor)


def test_convnext_unfreeze_prefixes_match_real_params():
    from src.extractors.convnext import ConvNeXtExtractor

    cls = ConvNeXtExtractor
    assert list(cls.unfreeze_prefixes) == ["backbone.stages.3"]


def test_convnext_extract_returns_correct_shape(extractor):
    image = Image.new("RGB", (224, 224))
    embedding = extractor.extract(image)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (64,)
    assert embedding.dtype == np.float32


def test_convnext_unfreeze_prefix_is_present_in_model(extractor):
    matches = [
        name
        for name, _ in extractor.model.named_parameters()
        if name.startswith("backbone.stages.3")
    ]
    assert matches, (
        "unfreeze_prefixes references 'backbone.stages.3' but no parameter "
        "matches that prefix in the constructed model."
    )


def test_convnext_registered_under_convnext_base():
    import src.extractors  # noqa: F401
    from src.extractors.convnext import ConvNeXtExtractor
    from src.extractors.registry import get_extractor_class

    assert get_extractor_class("convnext_base") is ConvNeXtExtractor
