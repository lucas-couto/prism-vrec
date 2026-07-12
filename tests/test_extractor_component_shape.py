"""Per-extractor component-feature shape contract for ACF.

Each component-capable extractor must return ``(N, M, native_dim)`` from
``_forward_components`` (``M`` = spatial cells / patch tokens, confirmed
per backbone; under the v2 protocol the last dim is the backbone's NATIVE
dim — the shared projection is gone).  Import-guarded and
instantiation-based, mirroring ``tests/test_convnext_extractor.py``
(weights are downloaded on first run).
"""

from __future__ import annotations

import pytest
import torch

BATCH = 2

# (extractor name, backend module to importorskip, expected component count M)
CASES = [
    ("resnet50", "torchvision", 49),
    ("vit_b16", "timm", 196),
    ("cvt_13", "transformers", 196),
    ("convnext_base", "timm", 49),
    ("coatnet_0", "timm", 49),
    ("levit_256", "timm", 16),
    ("clip_vitb32", "open_clip", 49),
    ("dinov2_vitb14", "torch", 256),
]


@pytest.mark.slow
@pytest.mark.parametrize(("name", "backend", "expected_m"), CASES)
def test_component_shape_is_n_m_native_dim(name, backend, expected_m) -> None:
    pytest.importorskip(backend)
    from src.extractors.registry import get_extractor_class

    extractor = get_extractor_class(name)(device="cpu")
    assert extractor.supports_components is True

    images = torch.randn(BATCH, 3, 224, 224)
    with torch.no_grad():
        components = extractor._forward_components(images)

    assert components.shape == (BATCH, expected_m, extractor.native_dim)
    assert components.dtype == torch.float32


def test_every_component_capable_extractor_can_produce_components() -> None:
    """``supports_components=True`` must be backed by a real component path.

    The base :meth:`_forward_components` delegates to the backbone's
    ``forward_components``; an extractor either overrides the method or
    declares a ``backbone_cls`` whose class exposes ``forward_components``.
    """
    import src.extractors  # noqa: F401  (populate registry)
    from src.extractors.base import BaseExtractor
    from src.extractors.registry import get_extractor_class, registered_extractor_names

    for name in registered_extractor_names():
        cls = get_extractor_class(name)
        if not getattr(cls, "supports_components", False):
            continue
        overrides_method = cls._forward_components is not BaseExtractor._forward_components
        backbone = getattr(cls, "backbone_cls", None)
        backbone_has_components = backbone is not None and hasattr(backbone, "forward_components")
        assert overrides_method or backbone_has_components, (
            f"{name} advertises supports_components but has neither a "
            f"_forward_components override nor a backbone_cls exposing forward_components"
        )
