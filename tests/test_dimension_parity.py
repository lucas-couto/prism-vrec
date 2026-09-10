"""Dimension parity: every recommender spends the same ``common.total_dim``.

The budget is the COLLABORATIVE capacity: ``latent_dim = T`` for every
registered model, with a visual model's own dimensions alongside it
rather than taken out of it.  A comparison at a fixed ``T`` is then a
comparison of the visual mechanism, not of how many collaborative
factors each model was left with.  ``dim_split="half"`` (``T/2`` latent
+ ``T/2`` visual) is still supported and still tested here through a
dummy model, but no recommender registers it since 2026-09-09 -- it gave
VBPR half of BPR's collaborative factors at the same ``T``, which was
the leading explanation for VBPR trailing BPR (audit hypothesis H1).
The budget is resolved per model through ``RecommenderSpec.dim_split``
and guarded before any training starts.
"""

from __future__ import annotations

import pytest
import torch

from src.recommenders.base import BaseRecommender
from src.recommenders.hp_search import (
    DimensionParityError,
    assert_dimension_parity,
    get_hyperparam_grid,
    resolve_dimensions,
    sample_hyperparams,
)
from src.recommenders.registry import register_recommender


class _Dummy(BaseRecommender):
    def forward(self, u, p, n):
        zeros = torch.zeros_like(u, dtype=torch.float32)
        return zeros, zeros

    def predict(self, u, items):
        return torch.zeros(items.shape[0], dtype=torch.float32)


class _FirstChoiceTrial:
    def __init__(self) -> None:
        self.names: list[str] = []

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        return choices[0]


@pytest.fixture(autouse=True)
def _registry() -> None:
    register_recommender("parity_latent", _Dummy, requires_visual=False, dim_split="latent")
    register_recommender("parity_latent_visual", _Dummy, uses_visual_dim=True, dim_split="latent")
    register_recommender("parity_half", _Dummy, uses_visual_dim=True, dim_split="half")


class TestResolveDimensions:
    def test_should_spend_the_whole_budget_on_latent_for_latent_split(self) -> None:
        assert resolve_dimensions("parity_latent", 128) == {"latent_dim": 128}

    def test_should_mirror_the_budget_into_visual_dim_when_the_model_uses_it(self) -> None:
        assert resolve_dimensions("parity_latent_visual", 64) == {
            "latent_dim": 64,
            "visual_dim": 64,
        }

    def test_should_split_the_budget_fifty_fifty_for_half_split(self) -> None:
        assert resolve_dimensions("parity_half", 128) == {"latent_dim": 64, "visual_dim": 64}

    def test_should_refuse_an_odd_budget_for_half_split(self) -> None:
        with pytest.raises(DimensionParityError, match="50/50"):
            resolve_dimensions("parity_half", 65)

    def test_should_refuse_an_unknown_split_at_registration(self) -> None:
        with pytest.raises(ValueError, match="dim_split"):
            register_recommender("parity_bad", _Dummy, dim_split="thirds")


class TestGridExpansion:
    def test_should_expand_total_dim_into_model_dimensions_in_the_grid(self) -> None:
        cfg = {"common": {"total_dim": [64, 128], "learning_rate": [0.01], "l2_reg": [0.001]}}

        grid = get_hyperparam_grid("parity_half", cfg)

        assert [hp["total_dim"] for hp in grid] == [64, 128]
        assert [(hp["latent_dim"], hp["visual_dim"]) for hp in grid] == [(32, 32), (64, 64)]

    def test_should_keep_bpr_and_vbpr_on_the_same_total(self) -> None:
        cfg = {"common": {"total_dim": [128], "learning_rate": [0.01], "l2_reg": [0.001]}}

        (latent,) = get_hyperparam_grid("parity_latent", cfg)
        (half,) = get_hyperparam_grid("parity_half", cfg)

        assert latent["latent_dim"] == half["latent_dim"] + half["visual_dim"] == 128

    def test_should_expand_total_dim_when_sampling_from_lists(self) -> None:
        cfg = {"common": {"total_dim": [64, 128], "learning_rate": [0.01], "l2_reg": [0.001]}}
        trial = _FirstChoiceTrial()

        sampled = sample_hyperparams(trial, "parity_half", cfg)

        assert "total_dim" in trial.names
        assert "latent_dim" not in trial.names
        assert sampled == {
            "total_dim": 64,
            "learning_rate": 0.01,
            "l2_reg": 0.001,
            "latent_dim": 32,
            "visual_dim": 32,
        }

    def test_should_expand_total_dim_sampled_from_an_hp_space(self) -> None:
        cfg = {
            "common": {"total_dim": [64]},
            "parity_half": {"hp_space": {"total_dim": {"type": "categorical", "choices": [128]}}},
        }

        sampled = sample_hyperparams(_FirstChoiceTrial(), "parity_half", cfg)

        assert (sampled["latent_dim"], sampled["visual_dim"]) == (64, 64)


class TestParityGuard:
    def test_should_pass_on_a_schema_validated_config(self) -> None:
        """The config schema materialises latent_dim/visual_dim as None; that
        is not a direct declaration (caught by the pre-battery validation run)."""
        from src.utils.config_schema import validate_config

        cfg = validate_config({"common": {"total_dim": [64]}, "recommenders_enabled": []})

        assert cfg["common"]["latent_dim"] is None
        assert_dimension_parity(cfg)

    def test_should_pass_when_only_total_dim_is_declared(self) -> None:
        cfg = {"common": {"total_dim": [64]}, "recommenders_enabled": ["parity_half"]}

        assert_dimension_parity(cfg)

    def test_should_refuse_a_config_without_total_dim(self) -> None:
        with pytest.raises(DimensionParityError, match="total_dim is required"):
            assert_dimension_parity({"common": {"latent_dim": [64], "visual_dim": [64]}})

    def test_should_refuse_direct_dims_next_to_the_budget(self) -> None:
        cfg = {"common": {"total_dim": [64], "visual_dim": [128]}}

        with pytest.raises(DimensionParityError, match="visual_dim"):
            assert_dimension_parity(cfg)

    def test_should_refuse_a_model_block_that_sets_its_own_dimensions(self) -> None:
        cfg = {
            "common": {"total_dim": [64]},
            "recommenders_enabled": ["parity_half"],
            "parity_half": {"latent_dim": [256]},
        }

        with pytest.raises(DimensionParityError, match="parity_half"):
            assert_dimension_parity(cfg)

    def test_should_refuse_an_hp_space_that_samples_dimensions_directly(self) -> None:
        cfg = {
            "common": {"total_dim": [64]},
            "recommenders_enabled": ["parity_half"],
            "parity_half": {"hp_space": {"latent_dim": {"type": "categorical", "choices": [8]}}},
        }

        with pytest.raises(DimensionParityError, match="hp_space.latent_dim"):
            assert_dimension_parity(cfg)


class TestRegisteredModelsShareTheCollaborativeBudget:
    """The decision of 2026-09-09, pinned on the real registry.

    Every registered recommender must hold ``latent_dim = T``; a visual
    model adds ``visual_dim = T`` beside it.  If VBPR ever returns to
    the 50/50 split it silently competes against BPR-MF with half the
    collaborative capacity, which is the confound H1 named.
    """

    @pytest.mark.parametrize("model", ["bpr", "vbpr", "vnpr", "deepstyle", "avbpr", "acf"])
    def test_every_registered_model_spends_the_full_budget_on_latent_factors(
        self, model: str
    ) -> None:
        assert resolve_dimensions(model, 128)["latent_dim"] == 128

    @pytest.mark.parametrize("model", ["vbpr", "avbpr"])
    def test_the_visual_side_sits_beside_the_budget_not_inside_it(self, model: str) -> None:
        dims = resolve_dimensions(model, 128)

        assert dims == {"latent_dim": 128, "visual_dim": 128}

    def test_vbpr_holds_as_many_collaborative_factors_as_bpr(self) -> None:
        assert (
            resolve_dimensions("vbpr", 128)["latent_dim"]
            == resolve_dimensions("bpr", 128)["latent_dim"]
        )
