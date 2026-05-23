"""Fusion strategies — pluggable via the registry.

Built-in strategies are registered when :mod:`src.fusions.strategies`
is imported (which happens here).  Custom strategies dropped under
``plugins/fusions/`` are auto-discovered at the same time.
"""

from src.fusions import strategies  # noqa: F401
from src.fusions.auto_register import scan_user_fusion_strategies  # noqa: E402
from src.fusions.online import (
    AdaptiveGatedFusion,
    load_embedding,
    online_module_for,
)
from src.fusions.registry import (
    FusionSpec,
    get_fusion_spec,
    get_fusion_strategy,
    is_online_strategy,
    is_registered,
    iter_specs,
    register_fusion_strategy,
    registered_fusion_strategies,
)

scan_user_fusion_strategies()


__all__ = [
    "AdaptiveGatedFusion",
    "FusionSpec",
    "get_fusion_spec",
    "get_fusion_strategy",
    "is_online_strategy",
    "is_registered",
    "iter_specs",
    "load_embedding",
    "online_module_for",
    "register_fusion_strategy",
    "registered_fusion_strategies",
]
