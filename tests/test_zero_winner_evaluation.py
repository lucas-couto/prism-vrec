"""I02 — an all-zero winner is evaluable; an absent/corrupt winner fails loudly.

End-to-end on the real components: ``train_single_run`` (scripted
selection metric, real persistence) writes the ``_best.pt`` of an
all-zero run, then the evaluate step loads it with the real full-ranking
``Evaluator`` and produces one complete per-user cell.  A best file that
cannot be loaded raises ``BestCheckpointError`` instead of being skipped
or "repaired" with guessed hyperparameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.steps.evaluate as ev
import src.utils.training as training_mod
from src.evaluation.protocol import Evaluator
from src.steps.evaluate import _evaluate_cell, find_best_models
from src.utils.checkpoint import BestCheckpointError
from tests.test_selection_zero_winner import (
    N_ITEMS,
    N_USERS,
    TRAIN,
    VAL,
    ScriptedEvaluator,
    _best_path,
    _config,
    _run,
)

TEST = {0: {3}, 1: {4}, 2: {5}, 3: {0}}


@pytest.fixture
def scripted(monkeypatch):
    """Scripted SELECTION evaluator; the final-evaluation one stays real."""
    ScriptedEvaluator.script = []
    ScriptedEvaluator.snapshots = []
    monkeypatch.setattr(training_mod, "Evaluator", ScriptedEvaluator)
    return ScriptedEvaluator


def _seen() -> dict[int, set[int]]:
    seen = {u: set(items) for u, items in TRAIN.items()}
    for u, items in VAL.items():
        seen[u].update(items)
    return seen


def _train_zero_winner(tmp_path: Path, scripted) -> dict:
    scripted.script = [{"ndcg@10": 0.0}]
    value = _run(tmp_path, _config(tmp_path, patience=2))
    assert value == 0.0
    models = find_best_models("ds", results_dir=tmp_path / "results")
    assert [(m["model_name"], m["embedding_name"]) for m in models] == [("bpr", "none")]
    return models[0]


def _evaluator() -> Evaluator:
    return Evaluator(_seen(), TEST, N_ITEMS, k_values=[10], tiebreak_seed=1)


class TestZeroWinnerReachesEvaluation:
    def test_all_zero_training_yields_one_complete_cell(self, tmp_path, scripted) -> None:
        model_info = _train_zero_winner(tmp_path, scripted)

        per_user = _evaluate_cell(
            model_info,
            "ds",
            N_USERS,
            N_ITEMS,
            _evaluator(),
            str(tmp_path / "emb"),
            "cpu",
            per_user_out_dir=str(tmp_path / "results"),
            seed=1,
        )

        assert per_user is not None
        assert sorted(per_user["user_id"]) == sorted(TEST)
        metric_cols = [c for c in per_user.columns if "@" in c]
        assert metric_cols
        values = per_user[metric_cols].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        # Zero is a legitimate per-user value; nothing filters it out.
        assert (values >= 0).all()
        assert set(per_user["protocol"]) == {"full_ranking"}

    def test_run_step_records_the_zero_cell(self, tmp_path, scripted, monkeypatch) -> None:
        _train_zero_winner(tmp_path, scripted)
        _write_processed(tmp_path / "p" / "ds")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ev,
            "load_config",
            lambda: {
                "device": "cpu",
                "seed": 1,
                "paths": {
                    "data_processed": str(tmp_path / "p"),
                    "embeddings": str(tmp_path / "emb"),
                    "results": str(tmp_path / "results"),
                },
                "k_values": [10],
                "datasets": ["ds"],
                "recommenders_enabled": ["bpr"],
                "extractors_enabled": [],
            },
        )

        ev.run("frozen")

        tables = tmp_path / "results" / "tables"
        for target in ("frozen", "finetuned"):
            out = pd.read_csv(tables / f"ds_evaluation_{target}.csv")
            assert len(out) == len(TEST)
            assert set(out["model_name"]) == {"bpr"}
        done = pd.read_csv(tables / "ds_evaluation_done.csv")
        assert len(done) == 2


class TestUnloadableWinnerFails:
    def test_corrupt_best_raises_instead_of_skipping(self, tmp_path, scripted) -> None:
        model_info = _train_zero_winner(tmp_path, scripted)
        Path(model_info["path"]).write_bytes(b"not a checkpoint")

        with pytest.raises(BestCheckpointError, match="unreadable"):
            _evaluate_cell(model_info, "ds", N_USERS, N_ITEMS, _evaluator(), "emb", "cpu")

    def test_absent_best_raises(self, tmp_path) -> None:
        model_info = {
            "model_name": "bpr",
            "embedding_name": "none",
            "path": str(_best_path(tmp_path)),
        }

        with pytest.raises(BestCheckpointError, match="does not exist"):
            _evaluate_cell(model_info, "ds", N_USERS, N_ITEMS, _evaluator(), "emb", "cpu")

    def test_legacy_flat_state_dict_is_identified_not_guessed(self, tmp_path) -> None:
        path = _best_path(tmp_path)
        path.parent.mkdir(parents=True)
        torch.save({"user_embedding.weight": torch.zeros(N_USERS, 4)}, path)
        model_info = {"model_name": "bpr", "embedding_name": "none", "path": str(path)}

        with pytest.raises(BestCheckpointError, match="legacy flat state_dict"):
            _evaluate_cell(model_info, "ds", N_USERS, N_ITEMS, _evaluator(), "emb", "cpu")

    def test_run_step_propagates_corrupt_best(self, tmp_path, scripted, monkeypatch) -> None:
        model_info = _train_zero_winner(tmp_path, scripted)
        Path(model_info["path"]).write_bytes(b"")
        _write_processed(tmp_path / "p" / "ds")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ev,
            "load_config",
            lambda: {
                "device": "cpu",
                "seed": 1,
                "paths": {
                    "data_processed": str(tmp_path / "p"),
                    "embeddings": str(tmp_path / "emb"),
                    "results": str(tmp_path / "results"),
                },
                "k_values": [10],
                "datasets": ["ds"],
                "recommenders_enabled": ["bpr"],
                "extractors_enabled": [],
            },
        )

        with pytest.raises(BestCheckpointError):
            ev.run("frozen")

        assert not (tmp_path / "results" / "tables" / "ds_evaluation_done.csv").exists()


def _write_processed(base: Path) -> None:
    base.mkdir(parents=True)
    for name, inter in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        rows = [(u, i) for u, items in inter.items() for i in sorted(items)]
        pd.DataFrame(rows, columns=["user_idx", "item_idx"]).to_csv(
            base / f"{name}.csv", index=False
        )
    (base / "user2idx.json").write_text(json.dumps({str(u): u for u in range(N_USERS)}))
    (base / "item2idx.json").write_text(json.dumps({str(i): i for i in range(N_ITEMS)}))
