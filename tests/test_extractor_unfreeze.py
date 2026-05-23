"""Each fine-tunable built-in extractor declares its own unfreeze prefixes.

Adding a new fine-tunable extractor is a one-file change:
``unfreeze_prefixes`` lives on the extractor class itself.  This test
guards against accidentally re-introducing a centralised lookup table.
"""

from __future__ import annotations

import pytest

from src.extractors.base import BaseExtractor


def test_base_extractor_default_is_empty_list() -> None:
    assert BaseExtractor.unfreeze_prefixes == []


@pytest.mark.parametrize(
    "module_path, class_name, expected_prefixes",
    [
        ("src.extractors.resnet", "ResNet50Extractor", ["features.8"]),
        (
            "src.extractors.vit",
            "ViTExtractor",
            ["backbone.blocks.11", "backbone.blocks.10"],
        ),
        ("src.extractors.cvt", "CvTExtractor", ["backbone.stages.2"]),
        ("src.extractors.coatnet", "CoAtNetExtractor", ["backbone.stages.3"]),
        ("src.extractors.levit", "LeViTExtractor", ["backbone.stages.2"]),
        (
            "src.extractors.convnext",
            "ConvNeXtExtractor",
            ["backbone.stages.3"],
        ),
    ],
)
def test_finetunable_extractors_declare_prefixes(
    module_path: str,
    class_name: str,
    expected_prefixes: list[str],
) -> None:
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    assert issubclass(cls, BaseExtractor)
    assert list(cls.unfreeze_prefixes) == expected_prefixes


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("src.extractors.clip", "CLIPExtractor"),
        ("src.extractors.dinov2", "DINOv2Extractor"),
    ],
)
def test_foundation_extractors_inherit_empty_prefixes(
    module_path: str,
    class_name: str,
) -> None:
    """Foundation extractors keep the default empty list (head-only FT)."""
    try:
        module = __import__(module_path, fromlist=[class_name])
    except (ImportError, AttributeError):
        pytest.skip(f"{module_path} not importable in this environment")
        return
    cls = getattr(module, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} not found in {module_path}")
        return
    assert list(cls.unfreeze_prefixes) == []
