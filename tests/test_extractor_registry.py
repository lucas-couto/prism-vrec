"""Plugin-contract tests for the extractor registry.

The registry is the canonical extension point for new visual extractors;
breaking any of these tests is a breaking change for plugin authors.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.extractors.base import BaseExtractor
from src.extractors.registry import (
    get_extractor_class,
    is_registered,
    register_extractor,
    registered_extractor_names,
)


class _DummyExtractor(BaseExtractor):
    """Minimal subclass that satisfies the abstract contract."""

    unfreeze_prefixes = ["features.last"]

    def __init__(self, device: str = "cpu", output_dim: int = 16) -> None:
        super().__init__(device=device, output_dim=output_dim)

    def _build_model(self) -> Any:
        return None

    def _build_transform(self) -> Any:
        return None


def test_register_extractor_accepts_subclass() -> None:
    register_extractor("dummy_register_subclass", _DummyExtractor)
    assert is_registered("dummy_register_subclass")
    assert get_extractor_class("dummy_register_subclass") is _DummyExtractor


def test_register_extractor_accepts_factory() -> None:
    register_extractor("dummy_register_factory", lambda: _DummyExtractor)
    cls = get_extractor_class("dummy_register_factory")
    assert cls is _DummyExtractor


def test_register_extractor_rejects_non_subclass() -> None:
    class NotAnExtractor:
        pass

    with pytest.raises(TypeError, match="must subclass BaseExtractor"):
        register_extractor("dummy_register_invalid", NotAnExtractor)


def test_register_extractor_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="expected a class or callable"):
        register_extractor("dummy_register_garbage", 42)  # type: ignore[arg-type]


def test_get_extractor_class_unknown_name_lists_available() -> None:
    register_extractor("dummy_listing_canary", _DummyExtractor)
    with pytest.raises(KeyError) as excinfo:
        get_extractor_class("definitely-not-registered")
    message = str(excinfo.value)
    assert "definitely-not-registered" in message
    assert "dummy_listing_canary" in message


def test_registered_extractor_names_is_sorted() -> None:
    names = registered_extractor_names()
    assert names == sorted(names)


def test_builtin_extractors_register_themselves() -> None:
    """Importing src.extractors must populate the registry."""
    import src.extractors  # noqa: F401

    builtins = {
        "resnet50",
        "vit_b16",
        "cvt_13",
        "coatnet_0",
        "levit_256",
        "clip_vitb32",
        "dinov2_vitb14",
        "convnext_base",
    }
    registered = set(registered_extractor_names())
    missing = builtins - registered
    assert not missing, f"built-in extractors not registered: {missing}"
