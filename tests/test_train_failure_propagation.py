"""Failed or unaccounted training work must fail the step and the run (E02, Q11/Q12).

Before this change ``_run_grid`` / ``_run_optuna`` summarised failures in
the log and returned normally, so ``main.py`` exited zero and the run
manifest said ``ok`` (audit F04).  The tests drive the real
:class:`TrainingOrchestrator` (sequential and spawned pool) and real
Optuna workers, with only the job enumeration replaced by synthetic
cells; the top-level boundary is exercised in-process through
``main.run_cli`` and the manifest ``finish_run`` writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import main
from src.battery.manifest import BatteryManifest
from src.recommenders.hp_search import CellKey
from src.steps import train
from src.steps.train import TrainingJobsFailedError, _raise_if_work_failed
from src.utils.parallel import TrainingJob, TrainingOrchestrator
from tests.test_parallel_job_outcomes import attempts_of, run_non_oom_error, run_succeed


def run_fail_flagged(job: TrainingJob) -> float:
    """Succeed unless the job carries ``fail=True`` (the artifacts of the
    other jobs are the partial work the step must preserve)."""
    if job.hyperparams.get("fail"):
        return run_non_oom_error(job)
    return run_succeed(job)


class TestReconciliation:
    def test_should_pass_when_every_expected_unit_succeeded(self) -> None:
        results = [{"job_id": "a", "status": "ok"}, {"job_id": "b", "status": "ok"}]

        _raise_if_work_failed(results, ["a", "b"], id_key="job_id", unit="job")

    def test_should_name_failed_and_unaccounted_units(self) -> None:
        results = [
            {"job_id": "a", "status": "ok"},
            {"job_id": "b", "status": "error", "error": "boom"},
            {"job_id": "b", "status": "ok"},  # duplicate: first delivery wins
        ]

        with pytest.raises(TrainingJobsFailedError) as excinfo:
            _raise_if_work_failed(results, ["a", "b", "c"], id_key="job_id", unit="job")

        err = excinfo.value
        assert err.total == 3
        assert [(f["id"], f["status"]) for f in err.failures] == [
            ("b", "error"),
            ("c", "unaccounted"),
        ]
        assert "2 of 3 jobs did not succeed" in str(err)
        assert "b: error (boom)" in str(err)
        assert "c: unaccounted" in str(err)


def _grid_config(tmp_path: Path) -> dict:
    return {
        "device": "cpu",
        "seed": 7,
        # The YAML is the only control surface: one step, one condition.
        "pipeline": {
            "run_all": False,
            "start_from": "train",
            "stop_at": "train",
            "condition": "frozen",
        },
        "paths": {
            "data_processed": str(tmp_path / "processed"),
            "embeddings": str(tmp_path / "embeddings"),
            "results": str(tmp_path / "results"),
        },
        "recommenders_enabled": ["vbpr"],
        "telemetry": {"enabled": False},
    }


def _job(scratch: Path, tag: str, **hp: object) -> TrainingJob:
    return TrainingJob(
        dataset_name="synthetic",
        model_name="vbpr",
        embedding_name="resnet50",
        hyperparams={"tag": tag, **hp},
        n_users=4,
        n_items=8,
        embeddings_path=None,
        processed_dir=str(scratch),
        device="cpu",
    )


def _patch_grid(monkeypatch, jobs: list[TrainingJob], runner) -> None:
    """Replace enumeration only; the orchestrator and its accounting are real."""

    class _Injected(TrainingOrchestrator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs, job_runner=runner)

    monkeypatch.setattr(train, "build_job_list", lambda *a, **k: list(jobs))
    monkeypatch.setattr(train, "_cell_counts", lambda *a, **k: {"vbpr": 1})
    monkeypatch.setattr(train, "_resolve_model_names", lambda config: ["vbpr"])
    monkeypatch.setattr(train, "get_hyperparam_grid", lambda name, config: [{}])
    monkeypatch.setattr(train, "TrainingOrchestrator", _Injected)


class TestGridStep:
    def test_should_raise_after_all_jobs_ran_and_keep_the_completed_ones(
        self, tmp_path, monkeypatch
    ):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        good = [_job(scratch, f"ok-{i}") for i in range(2)]
        bad = _job(scratch, "bad", fail=True)
        _patch_grid(monkeypatch, [bad, *good], run_fail_flagged)

        with pytest.raises(TrainingJobsFailedError) as excinfo:
            train._run_grid("frozen", _grid_config(tmp_path), workers=1, sequential=True)

        assert [f["id"] for f in excinfo.value.failures] == [bad.job_id]
        assert excinfo.value.failures[0]["status"] == "error"
        assert excinfo.value.total == 3
        # Every job ran to its own outcome: the failure did not abort the batch.
        assert all(attempts_of(j) == 1 for j in [bad, *good])

    def test_should_return_normally_when_every_job_succeeded(self, tmp_path, monkeypatch):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        _patch_grid(monkeypatch, [_job(scratch, "ok")], run_succeed)

        train._run_grid("frozen", _grid_config(tmp_path), workers=1, sequential=True)


def _optuna_config(tmp_path: Path) -> dict:
    return {
        **_grid_config(tmp_path),
        "hp_search": {
            "strategy": "optuna",
            "optuna": {"n_trials": 1, "storage": None, "n_startup_trials": 1},
        },
        "common": {"total_dim": [16], "learning_rate": [0.01], "l2_reg": [0.0]},
    }


class TestParallelOptunaStep:
    def test_should_raise_when_spawned_cell_workers_report_failures(self, tmp_path, monkeypatch):
        # The processed directory does not exist: every trial fails while
        # reading ``train.csv`` inside a real spawned worker.
        cells = [(CellKey("synthetic", "vbpr", f"emb{i}"), 4, 8, None) for i in range(2)]
        monkeypatch.setattr(train, "_list_cells", lambda *a, **k: cells)
        monkeypatch.setattr(train, "_resolve_model_names", lambda config: ["vbpr"])

        with pytest.raises(TrainingJobsFailedError) as excinfo:
            train._run_optuna("frozen", _optuna_config(tmp_path), workers=2)

        err = excinfo.value
        assert err.unit == "cell"
        assert err.total == 2
        assert sorted(f["id"] for f in err.failures) == sorted(c[0].study_name() for c in cells)
        assert {f["status"] for f in err.failures} == {"error"}
        assert all("train.csv" in (f["error"] or "") for f in err.failures)


def _manifest_status(results_root: Path) -> str | None:
    manifests = list((results_root / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1, manifests
    return json.loads(manifests[0].read_text(encoding="utf-8"))["exit_status"]


class TestCliBoundary:
    """``main.run_cli`` on a synthetic config with the real grid step underneath."""

    def _install(self, tmp_path, monkeypatch, jobs, runner) -> dict:
        config = _grid_config(tmp_path)
        _patch_grid(monkeypatch, jobs, runner)
        monkeypatch.setattr(main, "load_config", lambda *a, **k: config)
        monkeypatch.setitem(
            main.STEP_FUNCTIONS,
            "train",
            lambda condition: train._run_grid(condition, config, workers=1, sequential=True),
        )
        monkeypatch.chdir(tmp_path)
        return config

    def test_should_exit_nonzero_and_record_error_when_a_required_job_failed(
        self, tmp_path, monkeypatch
    ):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        good = _job(scratch, "ok")
        bad = _job(scratch, "bad", fail=True)
        self._install(tmp_path, monkeypatch, [good, bad], run_fail_flagged)

        code = main.run_cli([])

        assert code == 1
        assert _manifest_status(tmp_path / "results") == "error"
        # Partial valid work survives: the good job's artifact is still there.
        assert attempts_of(good) == 1

    def test_should_exit_zero_and_record_ok_when_every_job_succeeded(self, tmp_path, monkeypatch):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        self._install(tmp_path, monkeypatch, [_job(scratch, "ok")], run_succeed)

        code = main.run_cli([])

        assert code == 0
        assert _manifest_status(tmp_path / "results") == "ok"


class TestBatteryBoundary:
    def _run(self, tmp_path, monkeypatch, states: list[str]) -> int:
        import src.battery.runner as runner

        manifest = BatteryManifest(path=tmp_path / "manifest.json")
        for i, state in enumerate(states):
            manifest.set_state(f"cell-{i}", state)
        monkeypatch.setattr(runner, "run_battery", lambda *a, **k: manifest)
        monkeypatch.setattr(main, "load_config", lambda *a, **k: _grid_config(tmp_path))
        return main.run_cli(["--battery"])

    def test_should_exit_nonzero_when_the_battery_manifest_has_failed_cells(
        self, tmp_path, monkeypatch
    ):
        assert self._run(tmp_path, monkeypatch, ["done", "failed", "pending"]) == 1

    def test_should_exit_zero_when_every_battery_cell_is_done(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, ["done", "done"]) == 0
