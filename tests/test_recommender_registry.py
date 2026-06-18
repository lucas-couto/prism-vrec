"""Plugin-contract tests for the recommender registry."""

from __future__ import annotations

import pytest
import torch

from src.recommenders.base import BaseRecommender
from src.recommenders.registry import (
    RecommenderSpec,
    get_recommender_class,
    get_recommender_spec,
    is_registered,
    register_recommender,
    registered_recommender_names,
)


class _DummyRecommender(BaseRecommender):
    """Minimal subclass that satisfies the abstract contract."""

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(user_ids, dtype=torch.float32)
        return zeros, zeros

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(item_ids.shape[0], dtype=torch.float32)


def test_register_recommender_sets_defaults() -> None:
    register_recommender("test_dummy_defaults", _DummyRecommender)
    spec = get_recommender_spec("test_dummy_defaults")
    assert isinstance(spec, RecommenderSpec)
    assert spec.cls is _DummyRecommender
    assert spec.priority == 5
    assert spec.requires_visual is True
    assert spec.uses_visual_dim is False
    assert spec.extra_hyperparam_keys == ()
    assert spec.requires_components is False


def test_register_recommender_full_metadata() -> None:
    register_recommender(
        "test_dummy_full",
        _DummyRecommender,
        priority=1,
        requires_visual=False,
        uses_visual_dim=True,
        extra_hyperparam_keys=("alpha", "beta"),
        requires_components=True,
    )
    spec = get_recommender_spec("test_dummy_full")
    assert spec.priority == 1
    assert spec.requires_visual is False
    assert spec.uses_visual_dim is True
    assert spec.extra_hyperparam_keys == ("alpha", "beta")
    assert spec.requires_components is True


def test_register_recommender_rejects_non_subclass() -> None:
    class NotARecommender:
        pass

    with pytest.raises(TypeError, match="must subclass BaseRecommender"):
        register_recommender("test_dummy_invalid", NotARecommender)


def test_register_recommender_rejects_non_class() -> None:
    with pytest.raises(TypeError, match="must be a class"):
        register_recommender(
            "test_dummy_garbage",
            "not-a-class",  # type: ignore[arg-type]
        )


def test_get_recommender_class_unknown_name_lists_available() -> None:
    register_recommender("test_listing_canary", _DummyRecommender)
    with pytest.raises(KeyError) as excinfo:
        get_recommender_spec("definitely-unknown-recommender")
    message = str(excinfo.value)
    assert "definitely-unknown-recommender" in message
    assert "test_listing_canary" in message


def test_get_recommender_class_returns_subclass() -> None:
    register_recommender("test_class_lookup", _DummyRecommender)
    cls = get_recommender_class("test_class_lookup")
    assert cls is _DummyRecommender
    assert is_registered("test_class_lookup")


def test_registered_recommender_names_is_sorted() -> None:
    names = registered_recommender_names()
    assert names == sorted(names)


def test_builtin_recommenders_register_themselves() -> None:
    """Importing src.recommenders must populate the registry with built-ins."""
    import src.recommenders  # noqa: F401

    expected_subset = {"bpr", "vbpr", "avbpr", "vnpr", "deepstyle", "acf"}
    registered = set(registered_recommender_names())
    missing = expected_subset - registered
    assert not missing, f"built-in recommenders not registered: {missing}"


def test_acf_is_the_only_component_consuming_builtin() -> None:
    """Only ACF consumes component artifacts; all others use pooled embeddings."""
    component_models = {
        name
        for name in ("bpr", "vbpr", "vnpr", "deepstyle", "avbpr", "acf")
        if get_recommender_spec(name).requires_components
    }
    assert component_models == {"acf"}
