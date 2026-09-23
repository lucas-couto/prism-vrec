"""Paper properties that cut across recommenders.

Dimension parity at the MODEL level: ``resolve_dimensions(name, T)``
must leave every built-in recommender with the same COLLABORATIVE
capacity — ``T`` latent factors per user — so a comparison at a fixed
``T`` is a comparison of the visual mechanism rather than of how many
collaborative factors each model was left with.  A visual model's own
user-indexed table sits BESIDE that budget: VBPR/AVBPR add a visual
user vector of ``T`` (2026-09-09 decision) and VNPR adds one of ``D_v``
by construction (Niu et al. 2018).  Both are declared, not accidents.
The config-level guard (``total_dim`` only, direct dims refused) is
pinned in ``tests/test_dimension_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.recommenders import iter_specs
from src.recommenders.base import BaseRecommender
from src.recommenders.hp_search import resolve_dimensions
from src.recommenders.registry import RecommenderSpec

N_USERS, N_ITEMS, N_COMPONENTS, RAW_DIM = 5, 9, 3, 7
TOTAL_DIM = 8  # even: the (unregistered) half split must stay exact
BUILTIN = ("bpr", "vbpr", "avbpr", "deepstyle", "vnpr", "acf")
#: User parameters per user expected from ``dim_split``: ``T``
#: collaborative factors everywhere, plus whatever visual user vector the
#: model declares beside the budget.
EXPECTED_USER_DIMS = {
    "bpr": TOTAL_DIM,
    "vbpr": 2 * TOTAL_DIM,  # gamma_u in T plus theta_u in T
    "avbpr": 2 * TOTAL_DIM,  # mirrors VBPR
    "deepstyle": TOTAL_DIM,  # one p_u of dimension d = T
    "acf": TOTAL_DIM,  # U in k = T; the visual path is attention-only
    "vnpr": TOTAL_DIM + RAW_DIM,  # W_u in k = T plus W_v in D_v
}

#: The collaborative half of :data:`EXPECTED_USER_DIMS` — what parity is
#: actually about.  Every model must hold exactly ``T`` of these.
EXPECTED_VISUAL_USER_DIMS = {
    "bpr": 0,
    "vbpr": TOTAL_DIM,
    "avbpr": TOTAL_DIM,
    "deepstyle": 0,
    "acf": 0,
    "vnpr": RAW_DIM,
}
TRAIN = {0: {0, 1}, 1: {2, 3, 4}, 2: {5}, 3: {6, 7}}


def _visual(spec: RecommenderSpec) -> np.ndarray | None:
    if not spec.requires_visual:
        return None
    rng = np.random.default_rng(0)
    shape = (N_ITEMS, N_COMPONENTS, RAW_DIM) if spec.requires_components else (N_ITEMS, RAW_DIM)
    return rng.standard_normal(shape).astype("float32")


def _build(spec: RecommenderSpec, total_dim: int = TOTAL_DIM) -> BaseRecommender:
    """Model of ``spec`` sized by ``resolve_dimensions`` — no direct dims in the config."""
    config = {
        **resolve_dimensions(spec.name, total_dim),
        "att_hidden": 6,
        "max_history": 4,
        "history_seed": 3,
        "l2_reg": 0.0,
        "dropout": 0.0,
    }
    kwargs: dict = {}
    if getattr(spec.cls, "wants_history", False):
        kwargs["train_interactions"] = {u: set(s) for u, s in TRAIN.items()}
    if getattr(spec.cls, "wants_categories", False):
        kwargs["item_categories"] = None
    torch.manual_seed(0)
    return spec.cls(N_USERS, N_ITEMS, visual_embeddings=_visual(spec), config=config, **kwargs)


def _user_dims_per_user(model: BaseRecommender) -> int:
    return sum(
        module.embedding_dim
        for module in model.modules()
        if isinstance(module, nn.Embedding) and module.num_embeddings == N_USERS
    )


def _builtin_specs() -> list[RecommenderSpec]:
    specs = [spec for spec in iter_specs() if spec.name in BUILTIN]
    assert {spec.name for spec in specs} == set(BUILTIN)
    return specs


@pytest.mark.parametrize("spec", _builtin_specs(), ids=lambda s: s.name)
def test_user_parameters_per_user_match_the_dim_split_budget(spec: RecommenderSpec) -> None:
    model = _build(spec)

    assert _user_dims_per_user(model) == EXPECTED_USER_DIMS[spec.name]


@pytest.mark.parametrize("spec", _builtin_specs(), ids=lambda s: s.name)
def test_resolved_dimensions_sum_to_the_budget_for_the_half_split(spec: RecommenderSpec) -> None:
    dims = resolve_dimensions(spec.name, TOTAL_DIM)

    if spec.dim_split == "half":
        assert dims["latent_dim"] + dims["visual_dim"] == TOTAL_DIM
    else:
        assert dims["latent_dim"] == TOTAL_DIM


def test_every_built_in_spends_the_same_collaborative_budget() -> None:
    """Parity is about the COLLABORATIVE side: strip each model's declared
    visual user vector and every built-in must be left with exactly ``T``."""
    collaborative = {
        spec.name: _user_dims_per_user(_build(spec)) - EXPECTED_VISUAL_USER_DIMS[spec.name]
        for spec in _builtin_specs()
    }

    assert set(collaborative.values()) == {TOTAL_DIM}, collaborative


@pytest.mark.parametrize("total_dim", [4, 16])
def test_user_budget_scales_with_total_dim(total_dim: int) -> None:
    dims = {spec.name: _user_dims_per_user(_build(spec, total_dim)) for spec in _builtin_specs()}

    assert dims["bpr"] == dims["deepstyle"] == dims["acf"] == total_dim
    # The visual models carry their visual user vector beside the budget.
    assert dims["vbpr"] == dims["avbpr"] == 2 * total_dim
    assert dims["vnpr"] == total_dim + RAW_DIM
