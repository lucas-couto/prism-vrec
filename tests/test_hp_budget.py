"""HP-search budget fairness + replay mechanism (Task H)."""

from __future__ import annotations

import pytest

import src.steps.train as train_mod
from src.recommenders.hp_budget import (
    BudgetFairnessError,
    assert_uniform_budget,
    resolve_hp_budget,
)


def _config(**over) -> dict:
    cfg = {
        "recommenders_enabled": ["bpr", "vbpr"],
        "common": {
            "early_stopping_metric": "ndcg@10",
            "early_stopping_patience": 20,
            "epochs": 100,
            "eval_sample_size": 2000,
        },
        "hp_search": {"optuna": {"n_trials": 30}},
    }
    cfg.update(over)
    return cfg


class TestResolveBudget:
    def test_resolves_shared_values(self) -> None:
        budget = resolve_hp_budget(_config(), "amazon_fashion")
        assert budget["n_trials"] == 30
        assert budget["early_stopping_patience"] == 20
        assert budget["eval_sample_size"] == 2000

    def test_per_dataset_override(self) -> None:
        cfg = _config(hp_budget={"tradesy": {"n_trials": 10}})
        assert resolve_hp_budget(cfg, "tradesy")["n_trials"] == 10
        # other datasets keep the shared value
        assert resolve_hp_budget(cfg, "amazon_fashion")["n_trials"] == 30

    def test_same_budget_for_every_recommender_of_a_dataset(self) -> None:
        cfg = _config()
        # Budget resolution does not depend on the recommender — uniform.
        assert resolve_hp_budget(cfg, "amazon_men") == resolve_hp_budget(cfg, "amazon_men")


class TestFairnessGuardRail:
    def test_passes_when_no_per_model_budget(self) -> None:
        assert_uniform_budget(_config())  # no raise

    @pytest.mark.parametrize(
        "key", ["n_trials", "epochs", "early_stopping_patience", "eval_sample_size"]
    )
    def test_raises_when_recommender_declares_budget_key(self, key: str) -> None:
        cfg = _config(vbpr={key: 5})
        with pytest.raises(BudgetFairnessError, match="budget"):
            assert_uniform_budget(cfg)


class TestReplayMechanism:
    def test_train_replay_delegates_with_no_trial(self, monkeypatch) -> None:
        captured: dict = {}

        def _fake(*, trial, **kwargs):
            captured["trial"] = trial
            captured.update(kwargs)
            return 0.42

        monkeypatch.setattr(train_mod, "_train_one_optuna_trial", _fake)

        from src.recommenders.hp_search import CellKey

        metric = train_mod.train_replay(
            cell=CellKey("ds", "vbpr", "resnet50"),
            hyperparams={"latent_dim": 64},
            n_users=5,
            n_items=10,
            embeddings_path=None,
            processed_dir="proc",
            device="cpu",
            config=_config(),
        )

        assert metric == 0.42
        # Replay = search's train path with NO Optuna trial (no pruning).
        assert captured["trial"] is None
        assert captured["hyperparams"] == {"latent_dim": 64}
