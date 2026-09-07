"""R01 — tiny real-model battery cells: search then replay for every strategy.

Runs ``execute_cell`` end to end on a synthetic dataset with the real
training loop, real persistence and the real full-ranking evaluator, for
``fixed`` / ``grid`` / ``optuna``: the primary seed searches, a second
seed replays the selected effective configuration, and the replay's
final evaluation consumes the replay seed's own selected weights.
"""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
import torch

import src.steps.evaluate as evaluate_mod
from src.battery.cells import BatteryCell
from src.battery.execute import execute_cell
from src.battery.manifest import is_cell_complete
from src.evaluation.persistence import read_cell_artifact
from src.recommenders.hp_search import CellKey
from tests.test_folds_runner import _write_dataset

STUDY = CellKey("synthetic", "vbpr", "resnet50").study_name()


def _config(tmp_path: Path, strategy: str) -> dict:
    processed, embeddings = _write_dataset(tmp_path)
    return {
        "seed": 1,
        "seeds": [1, 2],
        "device": "cpu",
        "datasets": ["synthetic"],
        "recommenders_enabled": ["bpr", "vbpr"],
        "extractors_enabled": ["resnet50"],
        "fusion_strategies_enabled": [],
        "embedding_variants": "native",
        "pipeline": {"condition": "frozen"},
        "paths": {
            "data_processed": processed,
            "embeddings": embeddings,
            "results": str(tmp_path / "results"),
            "checkpoints": str(tmp_path / "checkpoints"),
        },
        "common": {
            "total_dim": 4 if strategy == "fixed" else [4, 8],
            "learning_rate": 0.05,
            "l2_reg": 0.0001,
            "epochs": 2,
            "batch_size": 8,
            "early_stopping_patience": 2,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": 1,
            "eval_sample_size": None,
        },
        "hp_search": {
            "strategy": strategy,
            "optuna": {
                "n_trials": 2,
                "sampler": "random",
                "pruner": "none",
                "storage": f"sqlite:///{tmp_path / 'optuna' / 'battery.db'}",
            },
        },
        "evaluation": {"protocol": "full_ranking"},
        "k_values": [5, 10],
    }


def _best_path(tmp_path: Path, seed: int) -> Path:
    return tmp_path / f"results_seed{seed}" / "models" / "synthetic" / "vbpr_resnet50_best.pt"


@pytest.fixture
def evaluated_checkpoints(monkeypatch) -> list[str]:
    """Record the ``_best.pt`` each final evaluation actually loads."""
    seen: list[str] = []
    real = evaluate_mod._evaluate_cell

    def _spy(model_info, *args, **kwargs):
        seen.append(model_info["path"])
        return real(model_info, *args, **kwargs)

    monkeypatch.setattr(evaluate_mod, "_evaluate_cell", _spy)
    return seen


@pytest.mark.parametrize("strategy", ["fixed", "grid", "optuna"])
def test_search_then_replay_evaluates_each_seed_on_its_own_selected_weights(
    tmp_path, strategy, evaluated_checkpoints
) -> None:
    cfg = _config(tmp_path, strategy)
    search = BatteryCell("synthetic", "resnet50", "vbpr", 1, "search")
    replay = BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay")

    search_result = execute_cell(search, cfg)
    replay_result = execute_cell(replay, cfg)

    assert search_result["strategy"] == strategy
    assert replay_result["strategy"] == strategy
    winner = torch.load(_best_path(tmp_path, 1), map_location="cpu", weights_only=False)
    replayed = torch.load(_best_path(tmp_path, 2), map_location="cpu", weights_only=False)
    assert replay_result["hyperparam_origin"]["hyperparams"] == winner["hyperparams"]
    assert replayed["hyperparams"] == winner["hyperparams"]
    assert evaluated_checkpoints == [str(_best_path(tmp_path, 1)), str(_best_path(tmp_path, 2))]
    assert any(
        not torch.equal(winner["model_state"][k], replayed["model_state"][k])
        for k in winner["model_state"]
    )
    assert is_cell_complete(search, tmp_path / "results")
    assert is_cell_complete(replay, tmp_path / "results")
    assert not (tmp_path / "results" / "models").exists()
    if strategy == "optuna":
        study = optuna.load_study(study_name=STUDY, storage=cfg["hp_search"]["optuna"]["storage"])
        assert len(study.trials) == 2
        assert replay_result["hyperparam_origin"]["provenance"]["best_trial"] == (
            study.best_trial.number
        )
    else:
        assert not (tmp_path / "optuna").exists()
    if strategy == "grid":
        assert search_result["n_configs"] == 2
        assert search_result["best_metric"] == winner["best_metric"]


def test_grid_search_selects_the_best_validation_configuration(tmp_path) -> None:
    cfg = _config(tmp_path, "grid")

    result = execute_cell(BatteryCell("synthetic", "none", "bpr", 1, "search"), cfg)

    grid = result["grid"]
    assert [g["hyperparams"]["latent_dim"] for g in grid] == [4, 8]
    best = max(grid, key=lambda g: g["best_metric"])
    winner = torch.load(
        tmp_path / "results_seed1" / "models" / "synthetic" / "bpr_none_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert winner["hyperparams"] == best["hyperparams"]
    assert winner["best_metric"] == best["best_metric"]


def test_replay_artifact_records_the_replay_seed(tmp_path) -> None:
    cfg = _config(tmp_path, "fixed")
    execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 1, "search"), cfg)
    execute_cell(BatteryCell("synthetic", "resnet50", "vbpr", 2, "replay"), cfg)

    per_user = tmp_path / "results" / "per_user" / "synthetic"
    artifacts = sorted(per_user.glob("*.csv.gz"))
    seeds = set()
    for path in artifacts:
        meta, df = read_cell_artifact(path)
        seeds.add(int(meta["seed"]))
        assert len(df) == 12
    assert seeds == {1, 2}
