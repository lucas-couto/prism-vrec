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
# ``dim_split`` divides the shared budget ``common.total_dim`` (T).
# Every recommender spends T on its COLLABORATIVE side, so a comparison
# against BPR-MF is a comparison of the visual mechanism rather than of
# the collaborative capacity: BPR-MF latent_dim = T; VBPR/AVBPR
# latent_dim = T with visual_dim = T alongside it; DeepStyle has a
# single d for p_u, q_i, s_i (Liu et al. 2017); VNPR's visual user
# vector lives in the image-feature space (D_v) by construction (Niu et
# al. 2018), so only its latent side counts; ACF scores in k dimensions
# and its visual path is attention-only.
#
# Until 2026-09-09 VBPR/AVBPR used ``dim_split="half"`` (T/2 latent +
# T/2 visual), which spent HALF the collaborative budget of every other
# model at the same T -- VBPR at T=128 held 64 latent factors against
# BPR's 128.  That accounting, not the visual term, was the leading
# explanation for VBPR trailing BPR (audit hypothesis H1).  The visual
# dimensions now sit alongside the budget instead of inside it, exactly
# as VNPR's already do.  ``"half"`` remains a supported value of
# ``dim_split``; nothing registers it any more.
register_recommender(
    "bpr",
    BPR,
    priority=0,
    requires_visual=False,
    uses_visual_dim=False,
    extra_hyperparam_keys=("l2_reg_item_pos", "l2_reg_item_neg"),
    dim_split="latent",
)
register_recommender(
    "vbpr",
    VBPR,
    priority=1,
    requires_visual=True,
    uses_visual_dim=True,
    extra_hyperparam_keys=("l2_reg_projection", "l2_reg_visual_bias"),
    dim_split="latent",
)
register_recommender(
    "vnpr",
    VNPR,
    priority=2,
    requires_visual=True,
    uses_visual_dim=False,
    extra_hyperparam_keys=("dropout",),
    dim_split="latent",
)
register_recommender(
    "deepstyle",
    DeepStyle,
    priority=3,
    requires_visual=True,
    uses_visual_dim=False,
    dim_split="latent",
)
register_recommender(
    "avbpr",
    AVBPR,
    priority=4,
    requires_visual=True,
    uses_visual_dim=True,
    extra_hyperparam_keys=("att_hidden", "l2_reg_projection", "l2_reg_visual_bias"),
    dim_split="latent",
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
    dim_split="latent",
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
