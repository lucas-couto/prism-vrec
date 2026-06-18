"""Recommender models for hybrid visual recommendation — pluggable via the registry.

Built-in recommenders register themselves at import time.  Custom
recommenders dropped under ``plugins/recommenders/`` are
auto-discovered via :mod:`src.recommenders.auto_register`.
"""

from src.recommenders.acf import ACF
from src.recommenders.avbpr import AVBPR
from src.recommenders.base import BaseRecommender
from src.recommenders.bpr import BPR
from src.recommenders.deepstyle import DeepStyle
from src.recommenders.registry import (
    RecommenderSpec,
    get_recommender_class,
    get_recommender_spec,
    is_registered,
    iter_specs,
    register_recommender,
    registered_recommender_names,
)
from src.recommenders.vbpr import VBPR
from src.recommenders.vnpr import VNPR

# Priority orders training so cheaper models finish first
# (BPR -> VBPR -> VNPR -> DeepStyle -> AVBPR).
register_recommender(
    "bpr",
    BPR,
    priority=0,
    requires_visual=False,
    uses_visual_dim=False,
)
register_recommender(
    "vbpr",
    VBPR,
    priority=1,
    requires_visual=True,
    uses_visual_dim=True,
)
register_recommender(
    "vnpr",
    VNPR,
    priority=2,
    requires_visual=True,
    uses_visual_dim=False,
    extra_hyperparam_keys=("hidden_layers",),
)
register_recommender(
    "deepstyle",
    DeepStyle,
    priority=3,
    requires_visual=True,
    uses_visual_dim=False,
    extra_hyperparam_keys=("style_dim",),
)
register_recommender(
    "avbpr",
    AVBPR,
    priority=4,
    requires_visual=True,
    uses_visual_dim=True,
    extra_hyperparam_keys=("att_hidden",),
)
# ACF consumes per-item component embeddings (3-D *_comp artifacts) and
# the user's training history; scheduled last (most expensive).
register_recommender(
    "acf",
    ACF,
    priority=5,
    requires_visual=True,
    uses_visual_dim=True,
    requires_components=True,
    extra_hyperparam_keys=("att_hidden", "max_history"),
)


from src.recommenders.auto_register import scan_user_recommenders  # noqa: E402

scan_user_recommenders()


__all__ = [
    "ACF",
    "BaseRecommender",
    "BPR",
    "VBPR",
    "VNPR",
    "DeepStyle",
    "AVBPR",
    "RecommenderSpec",
    "register_recommender",
    "get_recommender_spec",
    "get_recommender_class",
    "registered_recommender_names",
    "is_registered",
    "iter_specs",
]
