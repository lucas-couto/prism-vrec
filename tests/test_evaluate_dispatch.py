"""The evaluate step dispatches on ``folds.enabled`` (researcher decision, 2026-09-07).

``true`` runs the user-level K-fold protocol (the code path the removed
``--folds`` mode ran), ``false`` runs the single-split evaluator; never
both in one run.  Under K-fold the step materialises the battery tables
and the completion record the statistical step reconciles against
(R05) from the concatenated fold artifacts.  The resolved protocol is
execution metadata: recorded in the run manifest, printed by
``--show-plan``, absent from the scientific identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import main
import src.folds.runner as folds_runner
import src.steps.evaluate as ev
import src.steps.evaluate_kfold as ev_kfold
from src.battery.manifest import BatteryManifest, IncompleteRunError
from src.evaluation.persistence import read_cell_artifact
from src.recommenders.bpr import BPR
from src.recommenders.hp_search import CellKey
from src.steps import beyond_accuracy as ba_step
from src.steps import statistical as stat_step
from src.utils.config import set_config_override
from src.utils.evaluation_protocol import artifact_seed, resolve_evaluation_protocol
from src.utils.identity import canonical_digest
from src.utils.training import resolve_training_identity
from tests.test_folds_runner import N_USERS, _config, _write_dataset

_CELLS = (("bpr", "none"), ("vbpr", "resnet50"))


@pytest.fixture()
def synthetic_config(tmp_path: Path) -> dict:
    processed, embeddings = _write_dataset(tmp_path)
    cfg = _config(tmp_path, processed, embeddings)
    cfg["dataset_contracts"] = {"synthetic": {"expects_categories": False}}
    cfg["beyond_accuracy"] = {"enabled": True, "reference_embedding": "resnet50"}
    cfg["statistical"] = {
        "families": ["vs_baseline"],
        "bootstrap": {"enabled": True, "n_iterations": 10},
    }
    return cfg


@pytest.fixture()
def config_override():
    """Route every step's ``load_config()`` to the injected config."""
    yield set_config_override
    set_config_override(None)


class TestDispatch:
    def test_folds_enabled_runs_the_fold_runner_and_never_the_single_split(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cfg = {
            "device": "cpu",
            "datasets": ["synthetic"],
            "paths": {"results": str(tmp_path / "results")},
            "folds": {"enabled": True, "k": 3, "seed": 5},
        }
        calls: list[tuple] = []

        def _spy(config, results_dir, **kwargs):
            calls.append((config, Path(results_dir)))
            return BatteryManifest(path=tmp_path / "manifest.json")

        def _explode(*_a, **_k):
            raise AssertionError("the single-split evaluator must not run under K-fold")

        monkeypatch.setattr(ev, "load_config", lambda: cfg)
        monkeypatch.setattr(folds_runner, "run_folds", _spy)
        monkeypatch.setattr(ev, "_run_single_split", _explode)
        monkeypatch.setattr(ev, "find_best_models", _explode)

        ev.run("frozen")

        assert calls == [(cfg, tmp_path / "results")]

    def test_folds_disabled_runs_the_single_split_and_never_the_fold_runner(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cfg = {
            "device": "cpu",
            "datasets": ["synthetic"],
            "paths": {"results": str(tmp_path / "results")},
            "folds": {"enabled": False},
        }
        calls: list[tuple] = []

        def _explode(*_a, **_k):
            raise AssertionError("the K-fold runner must not run under the single split")

        monkeypatch.setattr(ev, "load_config", lambda: cfg)
        monkeypatch.setattr(folds_runner, "run_folds", _explode)
        monkeypatch.setattr(ev_kfold, "run_kfold", _explode)
        monkeypatch.setattr(ev, "_run_single_split", lambda config, root: calls.append(root))

        ev.run("frozen")

        assert calls == [tmp_path / "results"]

    def test_logs_which_protocol_was_chosen_and_why(self, tmp_path: Path, monkeypatch) -> None:
        cfg = {
            "device": "cpu",
            "datasets": ["synthetic"],
            "paths": {"results": str(tmp_path / "results")},
            "folds": {"enabled": True, "k": 2, "seed": 7},
        }
        lines: list[str] = []
        monkeypatch.setattr(ev, "load_config", lambda: cfg)
        monkeypatch.setattr(ev.logger, "info", lambda msg, *args: lines.append(msg % args))
        monkeypatch.setattr(
            folds_runner, "run_folds", lambda *a, **k: BatteryManifest(path=tmp_path / "m.json")
        )

        ev.run("frozen")

        assert any("kfold (k=2, partition seed=7; folds.enabled: true)" in m for m in lines)

    def test_an_incomplete_fold_run_fails_the_step_before_any_table_is_built(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        manifest = BatteryManifest(path=tmp_path / "manifest.json")
        manifest.set_state("cell-a", "done")
        manifest.set_state("cell-b", "failed")
        monkeypatch.setattr(folds_runner, "run_folds", lambda *a, **k: manifest)
        monkeypatch.setattr(
            ev_kfold, "materialise_tables", lambda *a, **k: pytest.fail("tables built")
        )

        with pytest.raises(IncompleteRunError, match="1 failed"):
            ev_kfold.run_kfold({"folds": {"enabled": True, "seed": 1}}, tmp_path / "results")


class TestRemovedFlag:
    def test_folds_flag_fails_naming_folds_enabled(self, capsys) -> None:
        code = main.run_cli(["--folds"])

        err = capsys.readouterr().err
        assert code == 2
        assert "--folds was removed" in err
        assert "folds.enabled" in err

    def test_folds_flag_is_gone_from_the_parser(self) -> None:
        known = {opt for action in main.build_parser()._actions for opt in action.option_strings}

        assert "--folds" not in known
        assert "--battery" in known

    @pytest.mark.parametrize(
        ("folds", "expected"),
        [
            ({"enabled": True, "k": 4, "seed": 9}, "kfold (k=4, partition seed=9"),
            ({"enabled": False}, "single_split (run seed=42"),
        ],
    )
    def test_show_plan_prints_the_evaluation_protocol(
        self, monkeypatch, capsys, folds, expected
    ) -> None:
        config = {"pipeline": {"run_all": True, "condition": "frozen"}, "folds": folds}
        monkeypatch.setattr(main, "load_config", lambda *a, **k: config)

        assert main.run_cli(["--show-plan"]) == 0

        assert f"Evaluation protocol: {expected}" in capsys.readouterr().out


class TestExecutionMetadata:
    def test_manifest_records_the_evaluation_protocol(self, tmp_path: Path) -> None:
        from src.utils.manifest import start_run

        run_dir = start_run(
            {"seed": 1, "device": "cpu", "folds": {"enabled": True, "k": 3, "seed": 8}},
            results_root=tmp_path / "runs",
        )

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["evaluation_protocol"] == {"mode": "kfold", "k": 3, "seed": 8}

    def test_manifest_records_the_single_split_when_folds_are_off(self, tmp_path: Path) -> None:
        from src.utils.manifest import start_run

        run_dir = start_run(
            {"seed": 11, "device": "cpu", "folds": {"enabled": False}},
            results_root=tmp_path / "runs",
        )

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["evaluation_protocol"] == {"mode": "single_split", "k": None, "seed": 11}

    def _identity(self, folds: dict) -> str:
        config = {"seed": 3, "common": {"epochs": 5, "batch_size": 8}, "folds": folds}
        payload = resolve_training_identity(
            model_cls=BPR,
            model_name="bpr",
            dataset_name="d",
            embedding_name="none",
            hyperparams={"latent_dim": 4},
            config=config,
        )
        assert "evaluation_protocol" not in json.dumps(payload)
        return canonical_digest(payload)

    def test_identity_digest_is_invariant_to_the_evaluation_protocol(self) -> None:
        assert self._identity({"enabled": False}) == self._identity(
            {"enabled": True, "k": 10, "seed": 99}
        )

    def test_folds_enabled_must_be_a_boolean(self) -> None:
        with pytest.raises(ValueError, match="folds.enabled"):
            resolve_evaluation_protocol({"folds": {"enabled": "yes"}})

    def test_artifact_seed_follows_the_protocol(self) -> None:
        assert artifact_seed({"seed": 1, "folds": {"enabled": True, "seed": 7}}) == 7
        assert artifact_seed({"seed": 1, "folds": {"enabled": False, "seed": 7}}) == 1


def _run_downstream(config: dict, override) -> tuple[Path, dict]:
    """evaluate -> beyond_accuracy -> statistical on ``config``; return tables + integrity."""
    override(config)
    ev.run("frozen")
    ba_step.run()
    stat_step.run("frozen")
    tables = Path(config["paths"]["results"]) / "tables"
    integrity = json.loads((tables / "synthetic_frozen_integrity.json").read_text())
    return tables, integrity


def _train_single_split_winners(config: dict) -> None:
    """Winners under ``results/models`` for the single-split evaluator to score.

    The same fixed-strategy route the battery executor takes: the pinned
    configuration is resolved by ``hp_source`` and trained once.
    """
    from src.recommenders.hp_source import resolve_cell_hyperparams
    from src.steps.train import train_replay

    for model, embedding in _CELLS:
        emb_path = None
        if embedding != "none":
            emb_path = str(Path(config["paths"]["embeddings"]) / "synthetic" / f"{embedding}.npy")
        origin = resolve_cell_hyperparams(
            config,
            dataset="synthetic",
            model_name=model,
            embedding_name=embedding,
            results_root=Path(config["paths"]["results"]),
        )
        train_replay(
            cell=CellKey("synthetic", model, embedding),
            hyperparams=origin.hyperparams,
            n_users=N_USERS,
            n_items=30,
            embeddings_path=emb_path,
            processed_dir=config["paths"]["data_processed"],
            device="cpu",
            config=config,
        )


class TestEndToEnd:
    def test_kfold_mode_feeds_beyond_accuracy_and_statistical(
        self, synthetic_config: dict, config_override
    ) -> None:
        results = Path(synthetic_config["paths"]["results"])

        tables, integrity = _run_downstream(synthetic_config, config_override)

        assert folds_runner.manifest_path(results).exists()
        artifacts = sorted((results / "per_user" / "synthetic").glob("*.csv.gz"))
        assert len(artifacts) == len(_CELLS)
        for path in artifacts:
            meta, records = read_cell_artifact(path)
            assert meta["fold"]["k"] == 2 and "index" not in meta["fold"]
            assert len(records) == N_USERS
        table = pd.read_csv(tables / "synthetic_evaluation_frozen.csv")
        assert len(table) == N_USERS * len(_CELLS)
        assert {"recall@5", "ndcg@10", "efd@5", "ild@10", "fold_policy"} <= set(table.columns)
        assert set(table["fold_policy"]) == {"kfold_k2"}
        done = pd.read_csv(tables / "synthetic_evaluation_done.csv")
        assert set(
            zip(done["target"], done["model_name"], done["embedding_name"], strict=True)
        ) == {
            ("frozen", "bpr", "none"),
            ("finetuned", "bpr", "none"),
            ("frozen", "vbpr", "resnet50"),
        }
        assert integrity["expected_source"] == "done_marker"
        assert integrity["cells_missing"] == []
        assert sorted(integrity["cells_completed"]) == ["bpr_none", "vbpr_resnet50"]
        assert (tables / "synthetic_frozen_pairwise.csv").exists()

    def test_kfold_mode_is_idempotent_on_a_second_evaluate(
        self, synthetic_config: dict, config_override, monkeypatch
    ) -> None:
        config_override(synthetic_config)
        ev.run("frozen")
        tables = Path(synthetic_config["paths"]["results"]) / "tables"
        before = pd.read_csv(tables / "synthetic_evaluation_frozen.csv")

        def _explode(*_a, **_k):
            raise AssertionError("cell re-executed although its fold artifact is valid")

        monkeypatch.setattr(folds_runner, "run_cell_folds", _explode)
        ev.run("frozen")

        after = pd.read_csv(tables / "synthetic_evaluation_frozen.csv")
        pd.testing.assert_frame_equal(before, after)

    def test_single_split_mode_feeds_beyond_accuracy_and_statistical(
        self, synthetic_config: dict, config_override
    ) -> None:
        synthetic_config["folds"]["enabled"] = False
        results = Path(synthetic_config["paths"]["results"])
        config_override(synthetic_config)
        _train_single_split_winners(synthetic_config)

        tables, integrity = _run_downstream(synthetic_config, config_override)

        assert not (results / "folds").exists()
        artifacts = sorted((results / "per_user" / "synthetic").glob("*.csv.gz"))
        assert len(artifacts) == len(_CELLS)
        for path in artifacts:
            meta, records = read_cell_artifact(path)
            assert meta["fold"] is None
            assert meta["seed"] == synthetic_config["seed"]
            assert len(records) == N_USERS
        table = pd.read_csv(tables / "synthetic_evaluation_frozen.csv")
        assert len(table) == N_USERS * len(_CELLS)
        assert {"recall@5", "ndcg@10", "efd@5", "ild@10"} <= set(table.columns)
        assert "fold_policy" not in table.columns
        assert integrity["expected_source"] == "done_marker"
        assert integrity["cells_missing"] == []
        assert sorted(integrity["cells_completed"]) == ["bpr_none", "vbpr_resnet50"]
        assert (tables / "synthetic_frozen_pairwise.csv").exists()
