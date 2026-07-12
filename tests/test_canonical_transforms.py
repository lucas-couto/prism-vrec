"""Canonical-preprocessing contract per backbone (v2, Mudança 1b).

Each backbone must use the recipe shipped with its weights.  Three
distinct normalisations coexist across the 8 backbones — a shared
generic transform was exactly the v1.x bug these tests pin against.

Marked slow: instantiating extractors loads real pretrained weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.extractors.registry import get_extractor_class

# White-pixel value on channel 0 identifies the normalisation family:
# (1.0 - mean_r) / std_r
_IMAGENET_WHITE = (1.0 - 0.485) / 0.229  # ≈ 2.249
_HALF_WHITE = (1.0 - 0.5) / 0.5  # = 1.000
_CLIP_WHITE = (1.0 - 0.48145466) / 0.26862954  # ≈ 1.930

CASES = [
    # (extractor, expected native dim, expected normalisation family)
    ("resnet50", 2048, _IMAGENET_WHITE),
    ("convnext_base", 1024, _IMAGENET_WHITE),
    ("vit_b16", 768, _HALF_WHITE),  # augreg2 tag: NOT ImageNet
    ("coatnet_0", 768, _HALF_WHITE),  # sw_in1k tag: NOT ImageNet
    ("levit_256", 512, _IMAGENET_WHITE),  # 512, not 256 — name is stage-1 width
    ("cvt_13", 384, _IMAGENET_WHITE),
    ("clip_vitb32", 512, _CLIP_WHITE),  # third normalisation family
    ("dinov2_vitb14", 768, _IMAGENET_WHITE),
]


@pytest.fixture(scope="module")
def white_image() -> Image.Image:
    return Image.new("RGB", (256, 256), (255, 255, 255))


@pytest.mark.slow
@pytest.mark.parametrize(("name", "native_dim", "white_value"), CASES)
def test_native_dim_and_canonical_normalisation(name, native_dim, white_value, white_image):
    extractor = get_extractor_class(name)(device="cpu")

    # Native dim read from the model, matching the known architecture value.
    assert extractor.native_dim == native_dim

    # Normalisation family probe: a pure-white image's channel-0 value
    # after the transform reveals which mean/std were applied.
    transformed = extractor.transform(white_image)
    assert transformed.shape[-2:] == (224, 224)
    assert transformed[0, 0, 0].item() == pytest.approx(white_value, abs=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize(("name", "native_dim", "_white"), CASES)
def test_pooled_output_is_native_and_metadata_complete(name, native_dim, _white):
    extractor = get_extractor_class(name)(device="cpu")

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (240, 200, 3), dtype="uint8"))
    tensor = extractor.transform(img).unsqueeze(0)
    with torch.no_grad():
        out = extractor.model(tensor)

    assert out.shape == (1, native_dim)

    meta = extractor.metadata()
    assert meta["native_dim"] == native_dim
    assert meta["extraction_point"] != "unspecified"
    assert meta["weights_id"] != "unspecified"
