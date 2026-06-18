"""Tests for the BaseRecommender 3-D component-buffer handling.

Guards the additive ``consumes_raw_components`` flag and the
``train_interactions`` constructor parameter without disturbing the
existing online-fusion (adaptive_gated) path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recommenders.base import BaseRecommender


class _Dummy(BaseRecommender):
    """Default model: 3-D buffers route to online fusion (two sources)."""

    def forward(self, user_ids, pos_item_ids, neg_item_ids):  # noqa: ANN001
        zeros = torch.zeros_like(user_ids, dtype=torch.float32)
        return zeros, zeros

    def predict(self, user_id, item_ids):  # noqa: ANN001
        return torch.zeros(item_ids.shape[0], dtype=torch.float32)


class _ComponentDummy(_Dummy):
    """Component-consuming model: 3-D buffers stay raw (no fusion)."""

    consumes_raw_components = True


def test_raw_components_flag_skips_online_fusion() -> None:
    components = np.zeros((10, 7, 8), dtype="float32")  # M=7 (not 2)

    model = _ComponentDummy(5, 10, components, {})

    assert model._online_fusion is None
    assert model.visual_features.dim() == 3
    assert model.visual_dim_raw == 8


def test_default_3d_buffer_still_instantiates_fusion_for_two_sources() -> None:
    stacked = np.zeros((10, 2, 8), dtype="float32")  # M=2 -> adaptive_gated

    model = _Dummy(5, 10, stacked, {})

    assert model._online_fusion is not None


def test_default_3d_buffer_rejects_non_two_sources() -> None:
    stacked = np.zeros((10, 3, 8), dtype="float32")  # M=3 -> invalid for fusion

    with pytest.raises(ValueError, match="2 source"):
        _Dummy(5, 10, stacked, {})


def test_two_dim_buffer_path_unchanged() -> None:
    pooled = np.zeros((10, 8), dtype="float32")

    model = _Dummy(5, 10, pooled, {})

    assert model._online_fusion is None
    assert model.visual_features.dim() == 2
    assert model.visual_dim_raw == 8


def test_train_interactions_defaults_to_none() -> None:
    model = _Dummy(5, 10, None, {})

    assert model.train_interactions is None


def test_train_interactions_is_stored_when_provided() -> None:
    history = {0: {1, 2}}

    model = _Dummy(5, 10, None, {}, train_interactions=history)

    assert model.train_interactions is history
