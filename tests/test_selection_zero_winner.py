"""I01 — the first finite selection observation is the winner, even at 0.0.

``train_single_run`` used to start ``best_metric`` at ``0.0`` and save a
trial-best only on strict improvement, and a missing selection key
defaulted to ``0.0``.  A legitimate all-zero run therefore finished
"successfully" with no ``_best.pt`` and silently vanished from the
evaluation, while a broken metric key looked exactly like a zero run.

These tests drive the real training loop (tiny BPR-MF, real
``CheckpointManager`` persistence in ``tmp_path``) with a scripted
selection evaluator so the observed metric sequence is under control.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import src.utils.training as training_mod
from src.recommenders.bpr import BPR
from src.utils.checkpoint import BestCheckpointError, CheckpointManager
from src.utils.training import SelectionMetricError, train_single_run

TRAIN = {0: {0, 1}, 1: {2, 3}, 2: {4}, 3: {5}}
VAL = {0: {2}, 1: {0}, 2: {1}, 3: {3}}
N_USERS, N_ITEMS = 4, 6
HYPERPARAMS = {"learning_rate": 0.01, "latent_dim": 4, "l2_reg": 0.0}


class ScriptedEvaluator:
    """Stands in for the selection ``Evaluator``: returns scripted metrics.

    Every call snapshots the model weights it was handed, so a test can
    check which epoch's weights ended up in the winner file.
    """

    script: list[dict] = []
    snapshots: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def evaluate(self, model, device: str = "cpu") -> dict:
        idx = len(type(self).snapshots)
        type(self).snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return type(self).script[min(idx, len(type(self).script) - 1)]


@pytest.fixture
def scripted(monkeypatch):
    ScriptedEvaluator.script = []
    ScriptedEvaluator.snapshots = []
    monkeypatch.setattr(training_mod, "Evaluator", ScriptedEvaluator)
    return ScriptedEvaluator


def _config(tmp_path: Path, *, epochs: int = 10, patience: int = 2) -> dict:
    return {
        "seed": 1,
        "paths": {"results": str(tmp_path / "results")},
        "common": {
            "epochs": epochs,
            "batch_size": 8,
            "early_stopping_patience": patience,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": 1,
        },
    }


def _run(tmp_path: Path, config: dict, **kwargs) -> float:
    return train_single_run(
        model_cls=BPR,
        model_name="bpr",
        n_users=N_USERS,
        n_items=N_ITEMS,
        visual_embeddings=None,
        train_interactions=TRAIN,
        selection_interactions=VAL,
        hyperparams=HYPERPARAMS,
        config=config,
        checkpoint_mgr=CheckpointManager(str(tmp_path / "ckpt")),
        dataset_name="ds",
        embedding_name="none",
        device="cpu",
        **kwargs,
    )


def _best_path(tmp_path: Path) -> Path:
    return tmp_path / "results" / "models" / "ds" / "bpr_none_best.pt"


def _assert_no_artifacts(tmp_path: Path) -> None:
    assert not _best_path(tmp_path).exists()
    assert list((tmp_path / "results").rglob("*_trial_*.pt")) == []
    assert list((tmp_path / "ckpt").rglob("*.pt")) == []


def _assert_same_weights(state: dict, snapshot: dict) -> None:
    for key, value in snapshot.items():
        assert torch.equal(state[key], value), f"weights diverge at {key}"


class TestFirstObservationWins:
    def test_zero_then_zero_saves_first_epoch_and_still_early_stops(self, tmp_path, scripted):
        scripted.script = [{"ndcg@10": 0.0}]

        value = _run(tmp_path, _config(tmp_path, patience=2))

        assert value == 0.0
        payload = torch.load(_best_path(tmp_path), map_location="cpu", weights_only=False)
        assert payload["best_metric"] == 0.0
        _assert_same_weights(payload["model_state"], scripted.snapshots[0])
        # First observation (epoch 0) then two non-improving evaluations.
        assert len(scripted.snapshots) == 3

    def test_zero_then_positive_winner_is_first_positive_epoch(self, tmp_path, scripted):
        scripted.script = [{"ndcg@10": 0.0}, {"ndcg@10": 0.2}, {"ndcg@10": 0.2}]

        value = _run(tmp_path, _config(tmp_path, patience=2))

        assert value == 0.2
        payload = torch.load(_best_path(tmp_path), map_location="cpu", weights_only=False)
        assert payload["best_metric"] == 0.2
        _assert_same_weights(payload["model_state"], scripted.snapshots[1])
        # Tie at epoch 2 keeps the earlier winner and advances patience.
        assert len(scripted.snapshots) == 4


class TestInvalidObservationsFail:
    def test_absent_metric_key_raises_and_leaves_no_artifact(self, tmp_path, scripted):
        scripted.script = [{"recall@10": 0.1}]

        with pytest.raises(SelectionMetricError, match="ndcg@10"):
            _run(tmp_path, _config(tmp_path))

        _assert_no_artifacts(tmp_path)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_metric_raises(self, tmp_path, scripted, bad):
        scripted.script = [{"ndcg@10": bad}]

        with pytest.raises(SelectionMetricError, match="finite"):
            _run(tmp_path, _config(tmp_path))

        _assert_no_artifacts(tmp_path)

    @pytest.mark.parametrize("bad", [[0.1], "0.1", None, True])
    def test_non_scalar_metric_raises(self, tmp_path, scripted, bad):
        scripted.script = [{"ndcg@10": bad}]

        with pytest.raises(SelectionMetricError, match="scalar"):
            _run(tmp_path, _config(tmp_path))

        _assert_no_artifacts(tmp_path)

    def test_schedule_without_validation_raises(self, tmp_path, scripted):
        scripted.script = [{"ndcg@10": 0.0}]

        with pytest.raises(SelectionMetricError, match="no validation"):
            _run(tmp_path, _config(tmp_path, epochs=0))

        _assert_no_artifacts(tmp_path)


class _PruneAtSecondReport:
    def __init__(self) -> None:
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return len(self.reports) >= 2


class TestPrunedTrialLeavesNoWinner:
    def test_pruned_trial_never_promotes_partial_winner(self, tmp_path, scripted):
        optuna = pytest.importorskip("optuna")
        scripted.script = [{"ndcg@10": 0.0}, {"ndcg@10": 0.5}]

        with pytest.raises(optuna.TrialPruned):
            _run(tmp_path, _config(tmp_path), optuna_trial=_PruneAtSecondReport())

        _assert_no_artifacts(tmp_path)


class TestPromotionRequiresLoadableWinner:
    def test_missing_trial_file_at_promotion_raises(self, tmp_path, scripted, monkeypatch):
        scripted.script = [{"ndcg@10": 0.3}]
        real_save = training_mod._save_trial_best

        def _save_then_lose(trial_path, *args, **kwargs):
            real_save(trial_path, *args, **kwargs)
            trial_path.unlink()

        monkeypatch.setattr(training_mod, "_save_trial_best", _save_then_lose)

        with pytest.raises(BestCheckpointError, match="does not exist"):
            _run(tmp_path, _config(tmp_path))

        assert not _best_path(tmp_path).exists()

    def test_corrupt_trial_file_at_promotion_raises(self, tmp_path, scripted, monkeypatch):
        scripted.script = [{"ndcg@10": 0.3}]
        real_save = training_mod._save_trial_best

        def _save_then_corrupt(trial_path, *args, **kwargs):
            real_save(trial_path, *args, **kwargs)
            trial_path.write_bytes(b"garbage")

        monkeypatch.setattr(training_mod, "_save_trial_best", _save_then_corrupt)

        with pytest.raises(BestCheckpointError, match="unreadable"):
            _run(tmp_path, _config(tmp_path))

        assert not _best_path(tmp_path).exists()
