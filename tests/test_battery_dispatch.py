"""R01 — the battery executor routes fixed / grid / Optuna faithfully (F09, Q16).

Dispatch is proven with spies on the two training entry points
(``train_replay`` and ``_optimize_one_cell``) and on study creation:
a grid cell must never open an Optuna study, a fixed cell must never
open one, and a replay must consume the primary seed's winner without
creating a study either.  Every replay trains under its own seed and
its own results/checkpoint namespace.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch

import src.battery.execute as ex
import src.recommenders.hp_search as hp_search
import src.steps.train as train_mod
from src.battery.cells import BatteryCell
from src.battery.execute import SearchOutcomeError
from src.recommenders.hp_search import CellKey, effective_hyperparams, get_hyperparam_grid
from src.recommenders.hp_source import WinnerResolutionError

N_USERS, N_ITEMS = 12, 30
CELL = CellKey("synthetic", "vbpr", "resnet50")


def _explode(*_a, **_k):
    raise AssertionError("this entry point must not be called on this dispatch path")


def _fixture(tmp_path: Path) -> None:
    proc = tmp_path / "processed" / "synthetic"
    emb = tmp_path / "embeddings" / "synthetic"
    proc.mkdir(parents=True)
    emb.mkdir(parents=True)
    (proc / "user2idx.json").write_text(json.dumps({str(i): i for i in range(N_USERS)}))
    (proc / "item2idx.json").write_text(json.dumps({str(i): i for i in range(N_ITEMS)}))
    np.save(emb / "resnet50.npy", np.zeros((N_ITEMS, 4), dtype=np.float32))


def _storage_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'optuna' / 'battery.db'}"


def _config(tmp_path: Path, strategy: str) -> dict:
    _fixture(tmp_path)
    total_dim = 8 if strategy == "fixed" else [4, 8]
    return {
        "seed": 1,
        "seeds": [1, 2],
        "device": "cpu",
        "datasets": ["synthetic"],
        "recommenders_enabled": ["bpr", "vbpr"],
        "hp_search": {
            "strategy": strategy,
            "optuna": {"n_trials": 2, "storage": _storage_url(tmp_path)},
        },
        "common": {
            "total_dim": total_dim,
            "learning_rate": [0.01],
            "l2_reg": 1e-4,
            "epochs": 2,
            "early_stopping_metric": "ndcg@10",
        },
        "paths": {
            "data_processed": str(tmp_path / "processed"),
            "embeddings": str(tmp_path / "embeddings"),
            "results": str(tmp_path / "results"),
            "checkpoints": str(tmp_path / "checkpoints"),
        },
    }


def _write_winner(tmp_path: Path, seed: int, hyperparams: dict, metric: float = 0.3) -> Path:
    path = tmp_path / f"results_seed{seed}" / "models" / "synthetic" / "vbpr_resnet50_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {},
            "hyperparams": hyperparams,
            "best_metric": metric,
            "n_users": N_USERS,
            "n_items": N_ITEMS,
            "selection_fingerprint": "fp",
        },
        path,
    )
    return path


@pytest.fixture
def spies(monkeypatch):
    """Record ``train_replay`` calls; forbid study creation everywhere."""
    trained: list[dict] = []
    evaluated: list[BatteryCell] = []
    monkeypatch.setattr(train_mod, "train_replay", lambda **k: trained.append(k) or 0.25)
    monkeypatch.setattr(ex, "_evaluate_one_cell", lambda cell, *a, **k: evaluated.append(cell))
    monkeypatch.setattr(hp_search, "create_study", _explode)
    monkeypatch.setattr(train_mod, "create_study", _explode)
    monkeypatch.setattr(optuna, "create_study", _explode)
    return trained, evaluated


class TestGridDispatch:
    def test_search_trains_every_grid_point_under_the_primary_seed_without_a_study(
        self, tmp_path, spies
    ) -> None:
        trained, evaluated = spies
        cfg = _config(tmp_path, "grid")
        grid = get_hyperparam_grid("vbpr", cfg)

        result = ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 1, "search"), cfg)

        assert [k["hyperparams"] for k in trained] == grid
        assert len(grid) == 2
        assert all(k["cell"] == CELL for k in trained)
        assert all(k["config"]["seed"] == 1 for k in trained)
        assert all(
            k["config"]["paths"]["results"] == str(tmp_path / "results_seed1") for k in trained
        )
        assert all(
            k["config"]["paths"]["checkpoints"] == str(tmp_path / "checkpoints_seed1")
            for k in trained
        )
        assert result["strategy"] == "grid"
        assert result["n_configs"] == 2
        assert len(evaluated) == 1
        assert not (tmp_path / "optuna").exists()

    def test_replay_trains_the_primary_winner_under_its_own_seed_without_a_study(
        self, tmp_path, spies
    ) -> None:
        trained, evaluated = spies
        cfg = _config(tmp_path, "grid")
        winner = get_hyperparam_grid("vbpr", cfg)[1]
        _write_winner(tmp_path, 1, winner)

        result = ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)

        assert len(trained) == 1
        assert trained[0]["hyperparams"] == winner
        assert trained[0]["config"]["seed"] == 2
        assert trained[0]["config"]["paths"]["results"] == str(tmp_path / "results_seed2")
        assert trained[0]["config"]["paths"]["checkpoints"] == str(tmp_path / "checkpoints_seed2")
        origin = result["hyperparam_origin"]
        assert origin["source"] == "search"
        assert origin["hyperparams"] == winner
        assert origin["best_metric"] == 0.3
        assert origin["provenance"]["strategy"] == "grid"
        assert origin["provenance"]["search_seed"] == 1
        assert result["strategy"] == "grid"
        assert len(evaluated) == 1
        assert not (tmp_path / "optuna").exists()

    def test_replay_without_a_winner_artifact_fails_explicitly(self, tmp_path, spies) -> None:
        trained, _ = spies
        cfg = _config(tmp_path, "grid")

        with pytest.raises(WinnerResolutionError, match="vbpr_resnet50_best.pt"):
            ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)

        assert trained == []

    def test_unequal_grid_sizes_are_reported_not_equalised(
        self, tmp_path, spies, caplog, monkeypatch
    ) -> None:
        cfg = _config(tmp_path, "grid")
        cfg["vbpr"] = {"l2_reg_visual_bias": [0.0, 1e-4]}
        ex._reset_grid_budget_warning_for_tests()
        # The project's loggers do not propagate to the root logger,
        # where caplog listens; let this one through for the assertion.
        monkeypatch.setattr(ex.logger, "propagate", True)

        with caplog.at_level("WARNING"):
            ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 1, "search"), cfg)

        assert any("UNEQUAL GRID-SEARCH BUDGETS" in r.getMessage() for r in caplog.records)
        assert len(spies[0]) == 4  # the declared space is trained as declared


class TestOptunaDispatch:
    def test_search_runs_the_trial_workflow_and_never_train_replay(
        self, tmp_path, monkeypatch
    ) -> None:
        cfg = _config(tmp_path, "optuna")
        calls: list[dict] = []
        summary = {
            "cell": CELL.study_name(),
            "status": "ok",
            "completed": 2,
            "pruned": 0,
            "best_value": 0.3,
            "best_params": {"total_dim": 8, "learning_rate": 0.01},
        }
        monkeypatch.setattr(
            train_mod, "_optimize_one_cell", lambda *a, **k: calls.append(k) or summary
        )
        monkeypatch.setattr(train_mod, "train_replay", _explode)
        monkeypatch.setattr(ex, "_evaluate_one_cell", lambda *a, **k: None)

        result = ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 1, "search"), cfg)

        assert len(calls) == 1
        assert calls[0]["config"]["seed"] == 1
        assert calls[0]["config"]["paths"]["checkpoints"] == str(tmp_path / "checkpoints_seed1")
        assert result["strategy"] == "optuna"
        assert result["search"]["completed"] == 2

    def test_search_whose_study_completed_no_trial_fails(self, tmp_path, monkeypatch) -> None:
        cfg = _config(tmp_path, "optuna")
        summary = {"cell": CELL.study_name(), "status": "ok", "completed": 0, "pruned": 2}
        monkeypatch.setattr(train_mod, "_optimize_one_cell", lambda *a, **k: summary)
        monkeypatch.setattr(ex, "_evaluate_one_cell", _explode)

        with pytest.raises(SearchOutcomeError, match="no COMPLETE trial"):
            ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 1, "search"), cfg)


#: One fixed search space for every injected trial (Optuna refuses a
#: parameter whose categorical choices change between trials).
_DISTRIBUTIONS = {
    "total_dim": optuna.distributions.CategoricalDistribution([4, 8]),
    "learning_rate": optuna.distributions.CategoricalDistribution([0.01]),
}


def _completed_trial(params: dict, value: float):
    distributions = {k: _DISTRIBUTIONS[k] for k in params}
    return optuna.trial.create_trial(params=params, distributions=distributions, value=value)


def _failed_trial(params: dict):
    distributions = {k: _DISTRIBUTIONS[k] for k in params}
    return optuna.trial.create_trial(
        params=params, distributions=distributions, state=optuna.trial.TrialState.FAIL
    )


def _seed_study(cfg: dict, trials: list) -> None:
    (Path(cfg["hp_search"]["optuna"]["storage"][len("sqlite:///") :]).parent).mkdir(
        parents=True, exist_ok=True
    )
    # ``optuna.study.create_study`` is the original; the spies fixture only
    # intercepts the ``optuna.create_study`` alias production code uses.
    study = optuna.study.create_study(
        study_name=CELL.study_name(),
        direction="maximize",
        storage=cfg["hp_search"]["optuna"]["storage"],
    )
    for trial in trials:
        study.add_trial(trial)


def _study_names(cfg: dict) -> list[str]:
    storage = cfg["hp_search"]["optuna"]["storage"]
    if not Path(storage[len("sqlite:///") :]).exists():
        return []
    return [s.study_name for s in optuna.get_all_study_summaries(storage=storage)]


class TestOptunaReplay:
    def test_replay_loads_the_existing_study_and_creates_none(self, tmp_path, spies) -> None:
        trained, evaluated = spies
        cfg = _config(tmp_path, "optuna")
        raw = {"total_dim": 8, "learning_rate": 0.01}
        effective = effective_hyperparams("vbpr", raw, cfg)
        _seed_study(cfg, [_completed_trial({"total_dim": 4, "learning_rate": 0.01}, 0.1)])
        _seed_study_more(cfg, [_completed_trial(raw, 0.3)])
        _write_winner(tmp_path, 1, effective)

        result = ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)

        assert len(trained) == 1
        assert trained[0]["hyperparams"] == effective
        assert trained[0]["config"]["seed"] == 2
        origin = result["hyperparam_origin"]
        assert origin["source"] == "search"
        assert origin["suggestion"] == raw
        assert origin["hyperparams"] == effective
        assert origin["provenance"]["strategy"] == "optuna"
        assert origin["provenance"]["best_trial"] == 1
        assert origin["provenance"]["n_completed"] == 2
        assert _study_names(cfg) == [CELL.study_name()]
        assert len(evaluated) == 1

    def test_replay_with_an_absent_study_fails_and_creates_none(self, tmp_path, spies) -> None:
        trained, _ = spies
        cfg = _config(tmp_path, "optuna")
        _write_winner(tmp_path, 1, effective_hyperparams("vbpr", {"total_dim": 8}, cfg))

        with pytest.raises(WinnerResolutionError, match="study"):
            ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)

        assert trained == []
        assert _study_names(cfg) == []

    def test_replay_with_a_failed_only_study_fails(self, tmp_path, spies) -> None:
        trained, _ = spies
        cfg = _config(tmp_path, "optuna")
        _seed_study(cfg, [_failed_trial({"total_dim": 8, "learning_rate": 0.01})])
        _write_winner(tmp_path, 1, effective_hyperparams("vbpr", {"total_dim": 8}, cfg))

        with pytest.raises(WinnerResolutionError, match="no COMPLETE trial"):
            ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)

        assert trained == []

    def test_replay_whose_winner_artifact_disagrees_with_the_study_fails(
        self, tmp_path, spies
    ) -> None:
        trained, _ = spies
        cfg = _config(tmp_path, "optuna")
        _seed_study(cfg, [_completed_trial({"total_dim": 8, "learning_rate": 0.01}, 0.3)])
        other = effective_hyperparams("vbpr", {"total_dim": 4, "learning_rate": 0.01}, cfg)
        _write_winner(tmp_path, 1, other, metric=0.2)

        with pytest.raises(WinnerResolutionError, match="disagree"):
            ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)

        assert trained == []

    def test_replay_without_a_winner_artifact_fails_even_with_a_study(
        self, tmp_path, spies
    ) -> None:
        cfg = _config(tmp_path, "optuna")
        _seed_study(cfg, [_completed_trial({"total_dim": 8, "learning_rate": 0.01}, 0.3)])

        with pytest.raises(WinnerResolutionError, match="vbpr_resnet50_best.pt"):
            ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)


def _seed_study_more(cfg: dict, trials: list) -> None:
    study = optuna.load_study(
        study_name=CELL.study_name(), storage=cfg["hp_search"]["optuna"]["storage"]
    )
    for trial in trials:
        study.add_trial(trial)


class TestFixedDispatch:
    @pytest.mark.parametrize("role", ["search", "replay"])
    def test_both_roles_train_the_pinned_configuration_in_their_own_namespace(
        self, tmp_path, spies, role
    ) -> None:
        trained, evaluated = spies
        cfg = _config(tmp_path, "fixed")
        seed = 1 if role == "search" else 2

        result = ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", seed, role), cfg)

        assert len(trained) == 1
        assert trained[0]["config"]["seed"] == seed
        assert trained[0]["config"]["paths"]["checkpoints"] == str(
            tmp_path / f"checkpoints_seed{seed}"
        )
        assert result["hyperparam_origin"]["source"] == "fixed"
        assert result["strategy"] == "fixed"
        assert len(evaluated) == 1
        assert not (tmp_path / "optuna").exists()

    def test_the_caller_config_is_never_mutated(self, tmp_path, spies) -> None:
        cfg = _config(tmp_path, "fixed")
        before = copy.deepcopy(cfg)

        ex.execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)

        assert cfg == before
