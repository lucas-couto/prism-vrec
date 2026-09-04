"""Tests for ``hp_search.strategy: fixed`` (no search, one run per cell).

Pins the contract the K-fold runner relies on: every hyperparameter
must resolve to exactly one value, each cell is trained exactly once
with ``trial=None``, and no Optuna study is ever created or read on
any path (dispatcher, ``train.run`` backend, battery executor).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.recommenders.hp_search as hp_search
from src.recommenders.hp_search import (
    STRATEGIES,
    CellKey,
    FixedHyperparamsError,
    get_fixed_hyperparams,
    get_hyperparam_grid,
    get_strategy,
    iter_cells,
)
from src.recommenders.registry import is_registered, register_recommender

_DUMMY = "test_fixed_dummy"


def _register_dummy() -> None:
    """Visual recommender with one extra key, so every branch is exercised."""
    if is_registered(_DUMMY):
        return
    import torch

    from src.recommenders.base import BaseRecommender

    class _Dummy(BaseRecommender):
        def forward(self, u, p, n):
            zeros = torch.zeros_like(u, dtype=torch.float32)
            return zeros, zeros

        def predict(self, u, items):
            return torch.zeros(items.shape[0], dtype=torch.float32)

    register_recommender(
        _DUMMY,
        _Dummy,
        priority=5,
        requires_visual=True,
        uses_visual_dim=True,
        extra_hyperparam_keys=("style_dim",),
    )


def _fixed_config(**overrides: object) -> dict:
    cfg = {
        "hp_search": {"strategy": "fixed"},
        "common": {"total_dim": 64, "learning_rate": [0.001], "l2_reg": 1e-4},
        _DUMMY: {"style_dim": [16]},
    }
    cfg.update(overrides)
    return cfg


class TestGetStrategy:
    def test_fixed_is_an_accepted_strategy(self) -> None:
        assert "fixed" in STRATEGIES
        assert get_strategy({"hp_search": {"strategy": "fixed"}}) == "fixed"

    def test_unknown_strategy_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown hp_search.strategy"):
            get_strategy({"hp_search": {"strategy": "bayes"}})


class TestGetFixedHyperparams:
    def test_scalars_and_unit_lists_resolve_to_one_configuration(self) -> None:
        _register_dummy()
        cfg = _fixed_config()

        hp = get_fixed_hyperparams(_DUMMY, cfg)

        assert hp["learning_rate"] == 0.001
        assert hp["l2_reg"] == 1e-4
        assert hp["style_dim"] == 16
        # total_dim is expanded into the model's own dimensions.
        assert hp["total_dim"] == 64
        assert hp["latent_dim"] == 64
        assert hp["visual_dim"] == 64

    def test_fixed_matches_the_single_grid_combination(self) -> None:
        _register_dummy()
        cfg = _fixed_config()

        assert get_hyperparam_grid(_DUMMY, cfg) == [get_fixed_hyperparams(_DUMMY, cfg)]

    def test_multivalued_keys_raise_naming_every_offender(self) -> None:
        _register_dummy()
        cfg = _fixed_config(
            common={"total_dim": [64, 128], "learning_rate": [0.001, 0.01], "l2_reg": 1e-4},
        )

        with pytest.raises(FixedHyperparamsError) as excinfo:
            get_fixed_hyperparams(_DUMMY, cfg)

        message = str(excinfo.value)
        assert "learning_rate" in message
        assert "total_dim" in message
        assert "l2_reg" not in message
        assert isinstance(excinfo.value, RuntimeError)

    def test_multivalued_extra_key_raises_too(self) -> None:
        _register_dummy()
        cfg = _fixed_config(**{_DUMMY: {"style_dim": [8, 16]}})

        with pytest.raises(FixedHyperparamsError, match="style_dim"):
            get_fixed_hyperparams(_DUMMY, cfg)


class TestIterCellsFixed:
    def test_objective_called_once_per_cell_with_trial_none(self, monkeypatch) -> None:
        _register_dummy()
        cfg = _fixed_config()
        cells = [CellKey("ds", _DUMMY, "resnet50"), CellKey("ds2", _DUMMY, "clip")]
        calls: list[tuple[CellKey, dict, object]] = []
        monkeypatch.setattr(hp_search, "create_study", _explode)

        def objective(cell, hp, trial):
            calls.append((cell, hp, trial))
            return 0.5

        out = list(iter_cells(cells, cfg, objective))

        assert [c for c, _, _ in calls] == cells
        assert all(trial is None for _, _, trial in calls)
        assert all(hp == get_fixed_hyperparams(_DUMMY, cfg) for _, hp, _ in calls)
        assert [m for _, _, m in out] == [0.5, 0.5]


def _explode(*_a, **_k):
    raise AssertionError("create_study must never be called under strategy 'fixed'")


def _train_config(tmp_path: Path) -> dict:
    return {
        "seed": 1,
        "device": "cpu",
        "datasets": ["synthetic"],
        "recommenders_enabled": ["bpr", "vbpr"],
        "hp_search": {"strategy": "fixed", "optuna": {"storage": "sqlite:///must/not/open.db"}},
        "common": {"total_dim": 64, "learning_rate": 0.001, "l2_reg": 1e-4},
        "paths": {
            "data_processed": str(tmp_path / "processed"),
            "embeddings": str(tmp_path / "embeddings"),
            "results": str(tmp_path / "results"),
        },
    }


class TestRunFixed:
    def test_trains_each_cell_once_via_train_replay_without_any_study(
        self, tmp_path, monkeypatch
    ) -> None:
        import src.steps.train as train_mod

        cfg = _train_config(tmp_path)
        cells = [
            (CellKey("synthetic", "bpr", "none"), 12, 30, None),
            (CellKey("synthetic", "vbpr", "resnet50"), 12, 30, "emb.npy"),
        ]
        trained: list[dict] = []
        monkeypatch.setattr(train_mod, "_list_cells", lambda *a, **k: list(cells))
        monkeypatch.setattr(train_mod, "train_replay", lambda **k: trained.append(k) or 0.1)
        monkeypatch.setattr(train_mod, "create_study", _explode)
        monkeypatch.setattr(train_mod, "_optimize_one_cell", _explode)
        monkeypatch.setattr(hp_search, "create_study", _explode)

        train_mod._run_fixed("frozen", cfg, workers=2, sequential=False)

        assert [(k["cell"], k["embeddings_path"]) for k in trained] == [
            (cells[0][0], None),
            (cells[1][0], "emb.npy"),
        ]
        assert trained[0]["hyperparams"] == get_fixed_hyperparams("bpr", cfg)
        assert trained[1]["hyperparams"] == get_fixed_hyperparams("vbpr", cfg)
        assert all(k["config"] is cfg and k["device"] == "cpu" for k in trained)
        assert not (tmp_path / "must").exists()

    def test_multivalued_key_fails_before_any_training(self, tmp_path, monkeypatch) -> None:
        import src.steps.train as train_mod

        cfg = _train_config(tmp_path)
        cfg["common"]["learning_rate"] = [0.001, 0.01]
        monkeypatch.setattr(train_mod, "_list_cells", _explode)
        monkeypatch.setattr(train_mod, "train_replay", _explode)

        with pytest.raises(FixedHyperparamsError, match="learning_rate"):
            train_mod._run_fixed("frozen", cfg, workers=1, sequential=True)

    def test_run_dispatches_fixed_to_run_fixed(self, tmp_path, monkeypatch) -> None:
        import src.steps.train as train_mod

        cfg = _train_config(tmp_path)
        seen: dict = {}
        monkeypatch.setattr(train_mod, "load_config", lambda: cfg)
        monkeypatch.setattr(train_mod, "assert_dimension_parity", lambda c: None)
        monkeypatch.setattr(
            "src.recommenders.hp_budget.assert_uniform_budget", lambda c: None, raising=False
        )
        monkeypatch.setattr(
            "src.steps.validate_features.gate_dataset_features",
            lambda *a, **k: None,
            raising=False,
        )
        monkeypatch.setattr(
            train_mod.CheckpointManager, "clear_all_training_checkpoints", lambda self: 0
        )
        monkeypatch.setattr(train_mod, "_run_grid", _explode)
        monkeypatch.setattr(train_mod, "_run_optuna", _explode)
        monkeypatch.setattr(train_mod, "_run_fixed", lambda *a, **k: seen.update(k))

        train_mod.run("frozen", workers=1, sequential=True)

        assert seen == {"workers": 1, "sequential": True}


def _battery_fixture(tmp_path: Path) -> None:
    proc = tmp_path / "processed" / "synthetic"
    emb = tmp_path / "embeddings" / "synthetic"
    proc.mkdir(parents=True)
    emb.mkdir(parents=True)
    (proc / "user2idx.json").write_text(json.dumps({str(i): i for i in range(12)}))
    (proc / "item2idx.json").write_text(json.dumps({str(i): i for i in range(30)}))
    np.save(emb / "resnet50.npy", np.zeros((30, 4), dtype=np.float32))


class TestExecuteCellFixed:
    @pytest.mark.parametrize("role", ["search", "replay"])
    def test_both_roles_train_replay_without_a_study(self, tmp_path, monkeypatch, role) -> None:
        import src.battery.execute as ex
        import src.steps.train as train_mod
        from src.battery.cells import BatteryCell

        _battery_fixture(tmp_path)
        cfg = _train_config(tmp_path)
        trained: list[dict] = []
        evaluated: list = []
        monkeypatch.setattr(train_mod, "train_replay", lambda **k: trained.append(k) or 0.1)
        monkeypatch.setattr(train_mod, "_optimize_one_cell", _explode)
        monkeypatch.setattr(hp_search, "create_study", _explode)
        monkeypatch.setattr(
            ex, "_evaluate_one_cell", lambda cell, c, *a, **k: evaluated.append(cell)
        )

        result = ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 7, role), cfg)

        assert len(trained) == 1
        assert trained[0]["cell"] == CellKey("synthetic", "vbpr", "resnet50")
        assert trained[0]["hyperparams"] == get_fixed_hyperparams("vbpr", cfg)
        assert trained[0]["config"]["seed"] == 7
        assert trained[0]["config"]["paths"]["results"] == str(tmp_path / "results") + "_seed7"
        assert len(evaluated) == 1
        assert result["role"] == role
        assert result["hyperparam_origin"] == {
            "source": "fixed",
            "hyperparams": get_fixed_hyperparams("vbpr", cfg),
            "reference": None,
            "best_metric": None,
        }
        assert not (tmp_path / "must").exists()
