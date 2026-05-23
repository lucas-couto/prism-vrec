"""Visual feature extractors — pluggable via the registry.

The eight built-in extractors are registered at import time.  Custom
extractors dropped under ``plugins/extractors/`` are auto-discovered
via :mod:`src.extractors.auto_register`.

Public API
----------

* :class:`BaseExtractor` — abstract base every extractor must subclass.
* :func:`register_extractor` — register a name -> class mapping.
* :func:`get_extractor_class` — resolve a name back to its class.
* :func:`registered_extractor_names` — list every registered name.
* ``EXTRACTOR_CLASSES`` — backwards-compat dict view of the registry.
"""

from src.extractors.base import BaseExtractor
from src.extractors.clip import CLIPExtractor
from src.extractors.coatnet import CoAtNetExtractor
from src.extractors.convnext import ConvNeXtExtractor
from src.extractors.cvt import CvTExtractor
from src.extractors.dinov2 import DINOv2Extractor
from src.extractors.levit import LeViTExtractor
from src.extractors.registry import (
    get_extractor_class,
    is_registered,
    register_extractor,
    registered_extractor_names,
)
from src.extractors.resnet import ResNet50Extractor
from src.extractors.vit import ViTExtractor

register_extractor("resnet50", ResNet50Extractor)
register_extractor("vit_b16", ViTExtractor)
register_extractor("cvt_13", CvTExtractor)
register_extractor("coatnet_0", CoAtNetExtractor)
register_extractor("levit_256", LeViTExtractor)
register_extractor("convnext_base", ConvNeXtExtractor)
register_extractor("clip_vitb32", CLIPExtractor)
register_extractor("dinov2_vitb14", DINOv2Extractor)


class _RegistryProxy(dict):
    """Read-only dict-like view of the extractor registry.

    Kept around so older code that did ``EXTRACTOR_CLASSES["resnet50"]``
    or ``"resnet50" in EXTRACTOR_CLASSES`` keeps working without
    snapshot drift after a plugin is registered later.
    """

    def __getitem__(self, key):
        return get_extractor_class(key)

    def __contains__(self, key) -> bool:
        return is_registered(key)

    def __iter__(self):
        return iter(registered_extractor_names())

    def __len__(self) -> int:
        return len(registered_extractor_names())

    def keys(self):
        return registered_extractor_names()

    def values(self):
        return [get_extractor_class(n) for n in registered_extractor_names()]

    def items(self):
        return [(n, get_extractor_class(n)) for n in registered_extractor_names()]

    def get(self, key, default=None):
        return get_extractor_class(key) if is_registered(key) else default


EXTRACTOR_CLASSES = _RegistryProxy()


from src.extractors.auto_register import scan_user_extractors  # noqa: E402

scan_user_extractors()


__all__ = [
    "BaseExtractor",
    "CLIPExtractor",
    "CoAtNetExtractor",
    "ConvNeXtExtractor",
    "CvTExtractor",
    "DINOv2Extractor",
    "LeViTExtractor",
    "ResNet50Extractor",
    "ViTExtractor",
    "EXTRACTOR_CLASSES",
    "register_extractor",
    "get_extractor_class",
    "registered_extractor_names",
    "is_registered",
]
