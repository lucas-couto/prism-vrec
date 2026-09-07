"""R03 — the resolved dataset budget is what training actually consumes (F16, Q16).

``resolve_hp_budget`` already honoured ``hp_budget[<dataset>]`` overrides,
but ``train_single_run`` read ``common:`` directly, so an override of
epochs / patience / selection metric / validation user sample changed
nothing at the training boundary while Optuna's trial count did change.
Each test here overrides one field and observes the value at the point
where it is consumed: the epoch loop (evaluations performed), the
selection ``Evaluator`` construction (user sample, cutoffs), the metric
lookup and the Optuna study — across every entry path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.utils.training as training_mod
from src.evaluation.protocol import Evaluator
from src.recommenders.bpr import BPR
from src.recommenders.hp_budget import (
    SELECTION_K_VALUES,
    UnsupportedSelectionMetricError,
    resolve_hp_budget,
)
from src.recommenders.hp_search import CellKey
from src.utils.checkpoint import CheckpointManager
from src.utils.training import train_single_run
from tests.test_folds_runner import N_ITEMS, N_USERS, _write_dataset

DATASET = "synthetic"


class RecordingEvaluator(Evaluator):
    """The real selection evaluator, recording what it was built with.

    ``constant`` replaces the metrics with a fixed dict so patience can
    be observed without depending on the tiny model's learning curve.
    """

    constructions: list[dict] = []
    evaluations: int = 0
    constant: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        type(self).constructions.append(dict(kwargs))
        super().__init__(*args, **kwargs)

    def evaluate(self, model, device: str = "cpu") -> dict:
        type(self).evaluations += 1
        metrics = super().evaluate(model, device=device)
        return metrics if type(self).constant is None else dict(type(self).constant)


@pytest.fixture
def recording(monkeypatch):
    RecordingEvaluator.constructions = []
    RecordingEvaluator.evaluations = 0
    RecordingEvaluator.constant = None
    monkeypatch.setattr(training_mod, "Evaluator", RecordingEvaluator)
    return RecordingEvaluator


@pytest.fixture
def consumed_metric(monkeypatch) -> list[str]:
    """Which selection key ``train_single_run`` actually looks up."""
    seen: list[str] = []
    real = training_mod._require_selection_metric

    def _spy(metrics, es_metric):
        seen.append(es_metric)
        return real(metrics, es_metric)

    monkeypatch.setattr(training_mod, "_require_selection_metric", _spy)
    return seen


def _config(tmp_path: Path, **budget_override) -> dict:
    processed, embeddings = _write_dataset(tmp_path)
    return {
        "seed": 3,
        "device": "cpu",
        "datasets": [DATASET],
        "recommenders_enabled": ["bpr"],
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
            "total_dim": 4,
            "learning_rate": 0.05,
            "l2_reg": 0.0001,
            "epochs": 6,
            "batch_size": 8,
            "early_stopping_patience": 10,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": 1,
            "eval_sample_size": None,
        },
        "hp_search": {
            "strategy": "fixed",
            "optuna": {"n_trials": 5, "sampler": "random", "pruner": "none", "storage": None},
        },
        "hp_budget": {DATASET: budget_override} if budget_override else {},
        "evaluation": {"protocol": "full_ranking"},
        "k_values": [5, 10],
        "folds": {
            "enabled": True,
            "k": 2,
            "seed": 7,
            "min_profile": 1,
            "fold_in": {"epochs": 1, "learning_rate": None, "batch_size": None},
        },
    }


def _interactions(processed: str, split: str) -> dict[int, set[int]]:
    df = pd.read_csv(Path(processed) / DATASET / f"{split}.csv")
    out: dict[int, set[int]] = {}
    for u, i in zip(df["user_idx"], df["item_idx"], strict=False):
        out.setdefault(int(u), set()).add(int(i))
    return out


def _train_direct(tmp_path: Path, cfg: dict) -> float:
    processed = cfg["paths"]["data_processed"]
    return train_single_run(
        model_cls=BPR,
        model_name="bpr",
        n_users=N_USERS,
        n_items=N_ITEMS,
        visual_embeddings=None,
        train_interactions=_interactions(processed, "train"),
        selection_interactions=_interactions(processed, "val"),
        hyperparams={"learning_rate": 0.05, "latent_dim": 4, "l2_reg": 0.0001},
        config=cfg,
        checkpoint_mgr=CheckpointManager(cfg["paths"]["checkpoints"]),
        dataset_name=DATASET,
        embedding_name="none",
        device="cpu",
    )


class TestResolver:
    def test_rejects_a_metric_the_selection_evaluator_does_not_produce(self) -> None:
        cfg = {"common": {"early_stopping_metric": "ndcg@5"}}

        with pytest.raises(UnsupportedSelectionMetricError, match="ndcg@5"):
            resolve_hp_budget(cfg, DATASET)

    @pytest.mark.parametrize("metric", ["hit_rate@10", "ndcg", "ndcg@x", "", 10])
    def test_rejects_malformed_or_unknown_metric_names(self, metric) -> None:
        cfg = {"hp_budget": {DATASET: {"early_stopping_metric": metric}}}

        with pytest.raises(UnsupportedSelectionMetricError):
            resolve_hp_budget(cfg, DATASET)

    @pytest.mark.parametrize("name", ["precision", "recall", "f1", "map", "ndcg"])
    def test_accepts_every_produced_metric_at_the_selection_cutoff(self, name) -> None:
        metric = f"{name}@{SELECTION_K_VALUES[0]}"

        budget = resolve_hp_budget({"common": {"early_stopping_metric": metric}}, DATASET)

        assert budget["early_stopping_metric"] == metric

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("epochs", -1),
            ("early_stopping_patience", 0),
            ("n_trials", 0),
            ("eval_sample_size", 0),
            ("epochs", "many"),
        ],
    )
    def test_rejects_out_of_range_budget_values(self, field, value) -> None:
        cfg = {"hp_budget": {DATASET: {field: value}}}

        with pytest.raises(ValueError, match=field):
            resolve_hp_budget(cfg, DATASET)

    def test_zero_epochs_is_an_empty_schedule_not_a_config_error(self) -> None:
        assert resolve_hp_budget({"common": {"epochs": 0}}, DATASET)["epochs"] == 0

    def test_override_values_are_normalised_to_their_types(self) -> None:
        cfg = {"hp_budget": {DATASET: {"epochs": "7", "eval_sample_size": "3"}}}

        budget = resolve_hp_budget(cfg, DATASET)

        assert budget["epochs"] == 7 and budget["eval_sample_size"] == 3


class TestTrainingBoundaryMatrix:
    """Direct ``train_single_run`` — the boundary the CLI single cell,
    the folds runner and every worker converge on."""

    def test_epochs_override_bounds_the_epoch_loop(self, tmp_path, recording) -> None:
        cfg = _config(tmp_path, epochs=2)

        _train_direct(tmp_path, cfg)

        assert recording.evaluations == 2  # common.epochs = 6 would give 6

    def test_patience_override_stops_the_loop(self, tmp_path, recording) -> None:
        cfg = _config(tmp_path, early_stopping_patience=2, epochs=50)
        recording.constant = {"ndcg@10": 0.1}

        _train_direct(tmp_path, cfg)

        # First observation + patience non-improving evaluations, not 10.
        assert recording.evaluations == 3

    def test_metric_override_is_the_key_looked_up(self, tmp_path, recording, consumed_metric):
        cfg = _config(tmp_path, early_stopping_metric="recall@10", epochs=1)

        _train_direct(tmp_path, cfg)

        assert consumed_metric == ["recall@10"]

    def test_user_sample_override_reaches_the_selection_evaluator(self, tmp_path, recording):
        cfg = _config(tmp_path, eval_sample_size=3, epochs=1)

        _train_direct(tmp_path, cfg)

        assert recording.constructions[0]["sample_size"] == 3
        assert recording.constructions[0]["k_values"] == list(SELECTION_K_VALUES)

    def test_unsupported_metric_fails_before_any_training(self, tmp_path, recording) -> None:
        cfg = _config(tmp_path, early_stopping_metric="ndcg@5")

        with pytest.raises(UnsupportedSelectionMetricError, match="ndcg@5"):
            _train_direct(tmp_path, cfg)

        assert recording.constructions == []
        assert not (tmp_path / "results").exists()

    def test_a_dataset_without_override_keeps_the_shared_values(self, tmp_path, recording):
        cfg = _config(tmp_path, epochs=2)
        cfg["hp_budget"] = {"another_dataset": cfg["hp_budget"][DATASET]}

        _train_direct(tmp_path, cfg)

        assert recording.evaluations == 6


class TestGridWorkerPath:
    def test_epochs_override_reaches_the_spawned_worker_config(
        self, tmp_path, recording, monkeypatch
    ) -> None:
        import src.steps.train as train_mod

        cfg = _config(tmp_path, epochs=2)
        cfg["hp_search"]["strategy"] = "grid"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("src.utils.config.load_config", lambda *a, **k: cfg)

        train_mod._run_grid("frozen", cfg, workers=1, sequential=True)

        assert recording.evaluations == 2


class TestOptunaPath:
    def test_trial_count_and_epochs_overrides_reach_the_study(self, tmp_path, recording) -> None:
        import src.steps.train as train_mod

        cfg = _config(tmp_path, n_trials=2, epochs=1)
        cfg["hp_search"]["strategy"] = "optuna"
        cell = CellKey(DATASET, "bpr", "none")

        summary = train_mod._optimize_one_cell(
            cell,
            N_USERS,
            N_ITEMS,
            None,
            config=cfg,
            processed_dir=cfg["paths"]["data_processed"],
            device="cpu",
        )

        assert summary["completed"] + summary["pruned"] == 2  # common n_trials = 5
        assert recording.evaluations == 2  # one evaluation per trial


class TestBatteryReplayPath:
    def test_epochs_override_reaches_the_replay_training(self, tmp_path, recording) -> None:
        from src.battery.cells import BatteryCell
        from src.battery.execute import execute_cell

        cfg = _config(tmp_path, epochs=2)
        cfg["seeds"] = [3, 4]

        result = execute_cell(BatteryCell(DATASET, "none", "bpr", 4, "replay"), cfg)

        assert recording.evaluations == 2
        assert result["budget"]["epochs"] == 2


class TestFoldsPath:
    def test_epochs_override_reaches_every_fold(self, tmp_path, recording) -> None:
        from src.folds.runner import run_folds

        cfg = _config(tmp_path, epochs=2)

        manifest = run_folds(cfg, cfg["paths"]["results"])

        assert manifest.summary()["done"] == 1
        assert recording.evaluations == 2 * cfg["folds"]["k"]


def test_selection_cutoff_is_the_single_declared_constant(tmp_path, recording) -> None:
    cfg = _config(tmp_path, epochs=1)

    _train_direct(tmp_path, cfg)

    assert recording.constructions[0]["k_values"] == [10]
    assert json.loads(json.dumps(list(SELECTION_K_VALUES))) == [10]
