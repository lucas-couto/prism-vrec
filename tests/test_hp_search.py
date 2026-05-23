"""Tests for the hyperparameter-search dispatcher.

The dispatcher in ``src/recommenders/hp_search.py`` is the single
entry point both the grid and Optuna backends share.  These tests
pin its public contract — strategy detection, grid materialisation
and Optuna sampling — without launching an actual Optuna study (the
study is already exercised by Optuna's own tests).
"""

from __future__ import annotations

from src.recommenders.hp_search import (
    CellKey,
    get_hyperparam_grid,
    get_strategy,
    has_hp_space,
    sample_hyperparams,
)
from src.recommenders.registry import register_recommender


def _register_dummy(name: str = "test_hp_dummy") -> None:
    """Register a minimal recommender just so the registry can answer."""
    import torch

    from src.recommenders.base import BaseRecommender

    class _Dummy(BaseRecommender):
        def forward(self, u, p, n):
            return torch.zeros_like(u, dtype=torch.float32), torch.zeros_like(
                u,
                dtype=torch.float32,
            )

        def predict(self, u, items):
            return torch.zeros(items.shape[0], dtype=torch.float32)

    register_recommender(
        name,
        _Dummy,
        priority=5,
        requires_visual=True,
        uses_visual_dim=True,
        extra_hyperparam_keys=("style_dim",),
    )


def test_strategy_default_is_grid() -> None:
    assert get_strategy({}) == "grid"


def test_strategy_reads_yaml_block() -> None:
    cfg = {"hp_search": {"strategy": "optuna"}}
    assert get_strategy(cfg) == "optuna"


def test_has_hp_space_detects_block() -> None:
    cfg = {"vbpr": {"hp_space": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2}}}}
    assert has_hp_space(cfg, "vbpr") is True
    assert has_hp_space(cfg, "bpr") is False


def test_has_hp_space_rejects_non_dict() -> None:
    cfg = {"vbpr": {"hp_space": "not-a-dict"}}
    assert has_hp_space(cfg, "vbpr") is False


def test_grid_includes_visual_dim_when_spec_uses_it() -> None:
    _register_dummy("test_hp_grid_visual")
    cfg = {
        "common": {
            "latent_dim": [16],
            "learning_rate": [1e-3],
            "l2_reg": [1e-4],
            "visual_dim": [64, 128],
        },
        "test_hp_grid_visual": {"style_dim": [32, 64]},
    }
    grid = get_hyperparam_grid("test_hp_grid_visual", cfg)
    assert len(grid) == 4  # 2 visual_dim × 2 style_dim
    assert all("visual_dim" in hp and "style_dim" in hp for hp in grid)


def test_grid_omits_visual_dim_when_not_used() -> None:
    _register_dummy("test_hp_grid_novisual")
    import torch

    from src.recommenders.base import BaseRecommender
    from src.recommenders.registry import register_recommender

    class _DummyNoVis(BaseRecommender):
        def forward(self, u, p, n):
            return torch.zeros_like(u, dtype=torch.float32), torch.zeros_like(
                u,
                dtype=torch.float32,
            )

        def predict(self, u, items):
            return torch.zeros(items.shape[0], dtype=torch.float32)

    register_recommender(
        "test_hp_grid_novisual",
        _DummyNoVis,
        priority=5,
        requires_visual=False,
        uses_visual_dim=False,
    )
    cfg = {"common": {"latent_dim": [16, 32]}}
    grid = get_hyperparam_grid("test_hp_grid_novisual", cfg)
    assert all("visual_dim" not in hp for hp in grid)


class _FakeTrial:
    """Minimal Optuna-like trial used to exercise sample_hyperparams.

    Records every suggestion call so tests can assert on the
    parameter space the dispatcher constructed.
    """

    def __init__(self) -> None:
        self.suggestions: list[tuple[str, str, tuple, dict]] = []

    def suggest_int(self, name, low, high, **kwargs):
        self.suggestions.append((name, "int", (low, high), kwargs))
        return low

    def suggest_float(self, name, low, high, **kwargs):
        self.suggestions.append((name, "float", (low, high), kwargs))
        return low

    def suggest_categorical(self, name, choices):
        self.suggestions.append((name, "categorical", tuple(choices), {}))
        return choices[0]


def test_sample_hyperparams_uses_hp_space_when_declared() -> None:
    _register_dummy("test_hp_sample_space")
    cfg = {
        "test_hp_sample_space": {
            "hp_space": {
                "latent_dim": {"type": "int", "low": 8, "high": 128, "log": True},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
                "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]},
            },
        },
    }
    trial = _FakeTrial()
    sampled = sample_hyperparams(trial, "test_hp_sample_space", cfg)

    names = {s[0]: s for s in trial.suggestions}
    assert names["latent_dim"][1] == "int"
    assert names["learning_rate"][1] == "float"
    assert names["optimizer"][1] == "categorical"
    assert "latent_dim" in sampled
    assert "optimizer" in sampled


def test_sample_hyperparams_falls_back_to_lists() -> None:
    """Without ``hp_space`` the sampler treats the legacy lists as
    categorical search dimensions — opt-in migration path."""
    _register_dummy("test_hp_sample_fallback")
    cfg = {
        "common": {
            "latent_dim": [16, 32, 64],
            "learning_rate": [1e-3],
            "l2_reg": [1e-4],
            "visual_dim": [64, 128],
        },
        "test_hp_sample_fallback": {"style_dim": [32, 64]},
    }
    trial = _FakeTrial()
    sampled = sample_hyperparams(trial, "test_hp_sample_fallback", cfg)

    suggested_names = [s[0] for s in trial.suggestions]
    assert "latent_dim" in suggested_names
    assert "visual_dim" in suggested_names  # uses_visual_dim=True
    assert "style_dim" in suggested_names  # extra_hyperparam_keys
    assert all(s[1] == "categorical" for s in trial.suggestions)
    assert sampled["latent_dim"] == 16  # _FakeTrial returns first choice


def test_cell_key_study_name_is_stable() -> None:
    cell = CellKey("amazon_fashion", "vbpr", "resnet50_D128")
    assert cell.study_name() == "amazon_fashion__vbpr__resnet50_D128"
