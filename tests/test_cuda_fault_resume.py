"""A CUDA context fault leaves the process as exit 75 and resumes one run.

Covers the pipeline's side of :mod:`src.supervisor`: classifying the
fault, stopping the job queue at the first poisoned job, reopening the
manifest and the sidecars, and the ``run_cli`` boundary end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import main
from src import supervisor
from src.utils import oom_recoveries, timing
from src.utils.cuda_faults import CudaContextLostError, cuda_context_lost, is_fatal_cuda_error
from src.utils.manifest import finish_run, resume_run, start_run
from src.utils.parallel import TrainingJob
from tests.test_parallel_job_outcomes import _count_attempt, _job, _sequential, attempts_of
from tests.test_train_failure_propagation import _grid_config, _patch_grid

XID8 = "CUDA error: the launch timed out and was terminated"


def run_xid8(job: TrainingJob) -> float:
    _count_attempt(job)
    raise torch.AcceleratorError(XID8)


def run_xid8_if_flagged(job: TrainingJob) -> float:
    if job.hyperparams.get("fault"):
        return run_xid8(job)
    _count_attempt(job)
    return 0.5


class TestClassification:
    def test_should_flag_a_launch_timeout(self) -> None:
        assert is_fatal_cuda_error(RuntimeError(XID8))

    def test_should_flag_a_fault_buried_in_the_cause_chain(self) -> None:
        try:
            try:
                raise RuntimeError(XID8)
            except RuntimeError as inner:
                raise ValueError("step failed") from inner
        except ValueError as outer:
            assert is_fatal_cuda_error(outer)

    @pytest.mark.parametrize(
        "exc",
        [
            torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            RuntimeError("CUDA error: device-side assert triggered"),
            ValueError("shape mismatch"),
        ],
    )
    def test_should_not_flag_errors_of_the_code_or_the_job(self, exc: BaseException) -> None:
        assert not is_fatal_cuda_error(exc)

    def test_should_not_initialise_cuda_when_probing(self) -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            pytest.skip("CUDA already initialised in this process")

        assert cuda_context_lost() is False


class TestOrchestratorStopsAtTheFault:
    def test_should_raise_after_the_first_poisoned_job(self, tmp_path: Path) -> None:
        jobs = [_job(tmp_path, f"j{i}") for i in range(6)]

        with pytest.raises(CudaContextLostError):
            _sequential(tmp_path, run_xid8).run(jobs)

        assert [attempts_of(job) for job in jobs] == [1, 0, 0, 0, 0, 0]

    def test_should_keep_the_jobs_that_ran_before_the_fault(self, tmp_path: Path) -> None:
        jobs = [_job(tmp_path, "ok"), _job(tmp_path, "bad", fault=True), _job(tmp_path, "left")]
        orchestrator = _sequential(tmp_path, run_xid8_if_flagged)

        with pytest.raises(CudaContextLostError):
            orchestrator.run(jobs)

        assert attempts_of(jobs[0]) == 1
        assert attempts_of(jobs[2]) == 0


class TestResumedRunDirectory:
    def test_should_reopen_the_manifest_and_log_the_restart(self, tmp_path: Path) -> None:
        run_dir = start_run({"seed": 1}, results_root=tmp_path)
        finish_run(run_dir, exit_status="cuda_fault")

        resume_run(run_dir, attempt=2, reason="CUDA context fault: launch timed out")

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["exit_status"] is None
        assert manifest["restarts"] == [
            {
                "attempt": 2,
                "resumed_at": manifest["restarts"][0]["resumed_at"],
                "previous_exit_status": "cuda_fault",
                "reason": "CUDA context fault: launch timed out",
            }
        ]

    def test_should_keep_earlier_step_and_cell_timings(self, tmp_path: Path) -> None:
        timing.reset_for_tests()
        timing.bind_run_dir(tmp_path)
        timing.record_step("extract", timing.now_iso(), 3.0)
        timing.record_cell("train", 1.0, dataset="d")
        timing.reset_for_tests()

        timing.bind_run_dir(tmp_path, resume=True)
        timing.record_step("train (frozen)", timing.now_iso(), 2.0)

        steps = json.loads((tmp_path / "steps.json").read_text())
        assert [s["name"] for s in steps] == ["extract", "train (frozen)"]
        assert len(json.loads((tmp_path / "step_timings.json").read_text())) == 1
        timing.reset_for_tests()

    def test_should_keep_earlier_oom_recoveries(self, tmp_path: Path) -> None:
        oom_recoveries.reset_for_tests()
        oom_recoveries.bind_run_dir(tmp_path)
        oom_recoveries.record_recovery(step="train", job="a", attempt=1)
        oom_recoveries.reset_for_tests()

        oom_recoveries.bind_run_dir(tmp_path, resume=True)
        oom_recoveries.record_recovery(step="train", job="b", attempt=1)

        rows = (tmp_path / oom_recoveries.FILENAME).read_text().splitlines()
        assert len(rows) == 3
        oom_recoveries.reset_for_tests()


class TestCliResumesOneRun:
    """Two ``run_cli`` calls stand in for two children of one supervisor."""

    def _install(self, tmp_path, monkeypatch, jobs, runner) -> None:
        config = _grid_config(tmp_path)
        _patch_grid(monkeypatch, jobs, runner)
        monkeypatch.setattr(main, "load_config", lambda *a, **k: config)
        monkeypatch.setitem(
            main.STEP_FUNCTIONS,
            "train",
            lambda condition: main.train._run_grid(condition, config, workers=1, sequential=True),
        )
        monkeypatch.chdir(tmp_path)

    def _child(self, monkeypatch, state: Path, attempt: int) -> None:
        timing.reset_for_tests()
        oom_recoveries.reset_for_tests()
        monkeypatch.setenv(supervisor.ENV_SUPERVISED, "1")
        monkeypatch.setenv(supervisor.ENV_STATE_DIR, str(state))
        monkeypatch.setenv(supervisor.ENV_ATTEMPT, str(attempt))

    def test_should_exit_75_then_finish_the_same_run(self, tmp_path, monkeypatch) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        state = tmp_path / "state"
        state.mkdir()
        good, bad = _job(scratch, "ok"), _job(scratch, "bad", fault=True)

        self._child(monkeypatch, state, attempt=1)
        self._install(tmp_path, monkeypatch, [good, bad], run_xid8_if_flagged)
        first = main.run_cli([])
        self._child(monkeypatch, state, attempt=2)
        self._install(tmp_path, monkeypatch, [bad], _run_succeed)
        second = main.run_cli([])

        manifests = list((tmp_path / "results" / "runs").glob("*/manifest.json"))
        assert (first, second) == (supervisor.EXIT_CUDA_FAULT, 0)
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())
        assert manifest["exit_status"] == "ok"
        assert [r["previous_exit_status"] for r in manifest["restarts"]] == ["cuda_fault"]
        assert "launch timed out" in (state / "fault.txt").read_text()
        timing.reset_for_tests()
        oom_recoveries.reset_for_tests()


def _run_succeed(job: TrainingJob) -> float:
    _count_attempt(job)
    return 0.5
