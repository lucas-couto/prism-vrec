"""Every submitted training job ends in exactly one terminal outcome (E01, Q11/Q12).

Fault injections from the execution SPEC: worker exit before the result
is published, CUDA OOM followed by success, repeated OOM exhaustion, a
non-OOM exception that merely *looks* like an OOM, and duplicate result
delivery.  The runner is injected through ``job_runner`` so the
orchestrator's real accounting, retry and reaping code runs on CPU
without a dataset; ``job.processed_dir`` points at a scratch directory
used as a cross-process attempt counter.
"""

from __future__ import annotations

import os
import queue
import signal
from pathlib import Path

import pytest
import torch

from src.utils.parallel import (
    MAX_OOM_RETRIES,
    TrainingJob,
    TrainingOrchestrator,
    _JobRegistry,
    _worker_fn,
)

_OOM = torch.cuda.OutOfMemoryError


def _count_attempt(job: TrainingJob) -> int:
    """Append one line to the job's attempt file; return the attempt number."""
    path = Path(job.processed_dir) / f"{job.job_id}.attempts"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()}\n")
    return len(path.read_text(encoding="utf-8").splitlines())


def attempts_of(job: TrainingJob) -> int:
    path = Path(job.processed_dir) / f"{job.job_id}.attempts"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


# Module-level runners: spawned workers unpickle them by import path.


def run_succeed(job: TrainingJob) -> float:
    _count_attempt(job)
    return 0.5


def run_oom_then_succeed(job: TrainingJob) -> float:
    if _count_attempt(job) <= int(job.hyperparams["oom_times"]):
        raise _OOM("CUDA out of memory (injected)")
    return 0.25


def run_always_oom(job: TrainingJob) -> float:
    _count_attempt(job)
    raise _OOM("CUDA out of memory (injected, irreducible)")


def run_non_oom_error(job: TrainingJob) -> float:
    _count_attempt(job)
    raise RuntimeError("CUDA out of memory: not the OOM exception type")


def run_exit_before_publish(job: TrainingJob) -> float:
    _count_attempt(job)
    os._exit(3)


def run_sigkill_before_publish(job: TrainingJob) -> float:
    _count_attempt(job)
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("unreachable")  # pragma: no cover


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


def _by_id(results: list[dict]) -> dict[str, dict]:
    return {r["job_id"]: r for r in results}


def _sequential(tmp_path: Path, runner) -> TrainingOrchestrator:
    return TrainingOrchestrator(
        n_workers=1, device="cpu", log_dir=str(tmp_path / "logs"), job_runner=runner
    )


def _pool(tmp_path: Path, runner, n_workers: int = 2) -> TrainingOrchestrator:
    return TrainingOrchestrator(
        n_workers=n_workers, device="cuda", log_dir=str(tmp_path / "logs"), job_runner=runner
    )


class TestSequentialPath:
    def test_should_retry_oom_once_and_succeed_on_the_second_attempt(self, tmp_path):
        job = _job(tmp_path, "oom-once", oom_times=1)

        results = _sequential(tmp_path, run_oom_then_succeed).run([job])

        assert attempts_of(job) == 2
        assert results == [
            {
                "job_id": job.job_id,
                "outcome": "succeeded",
                "attempts": 2,
                "status": "ok",
                "best_metric": 0.25,
            }
        ]

    def test_should_fail_after_exactly_max_oom_retries_plus_one_attempts(self, tmp_path):
        job = _job(tmp_path, "oom-always")

        results = _sequential(tmp_path, run_always_oom).run([job])

        assert attempts_of(job) == MAX_OOM_RETRIES + 1
        (result,) = results
        assert result["status"] == "oom"
        assert result["outcome"] == "failed"
        assert result["attempts"] == MAX_OOM_RETRIES + 1
        assert result["error_type"] == "OutOfMemoryError"

    def test_should_never_retry_an_exception_that_is_not_the_oom_type(self, tmp_path):
        job = _job(tmp_path, "non-oom")

        results = _sequential(tmp_path, run_non_oom_error).run([job])

        assert attempts_of(job) == 1
        (result,) = results
        assert result["status"] == "error"
        assert result["outcome"] == "failed"
        assert result["error_type"] == "RuntimeError"
        assert "not the OOM exception type" in result["error"]

    def test_should_return_exactly_one_outcome_per_submitted_job(self, tmp_path):
        jobs = [_job(tmp_path, f"ok-{i}") for i in range(3)]

        results = _sequential(tmp_path, run_succeed).run(list(jobs))

        assert sorted(r["job_id"] for r in results) == sorted(j.job_id for j in jobs)
        assert {r["status"] for r in results} == {"ok"}
        assert all(attempts_of(j) == 1 for j in jobs)


class TestDuplicateDelivery:
    def test_should_keep_one_outcome_when_a_result_is_delivered_twice(self, tmp_path):
        job = _job(tmp_path, "dup")
        registry = _JobRegistry([job])
        job_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()
        job_queue.put(job)
        job_queue.put(None)
        _worker_fn(0, job_queue, result_queue, 1, str(tmp_path / "logs"), run_succeed)
        first = result_queue.queue[0]
        result_queue.put(dict(first))
        result_queue.put({**first, "status": "error", "error": "stale"})

        TrainingOrchestrator._drain_sequential(registry, [job], result_queue)

        (outcome,) = registry.outcomes()
        assert outcome.status == "succeeded"
        assert outcome.attempt_count == 1
        assert registry.results()[0]["status"] == "ok"

    def test_should_ignore_a_second_message_after_a_worker_exit_was_accounted(self, tmp_path):
        job = _job(tmp_path, "late")
        registry = _JobRegistry([job])
        registry.fail(job.job_id, error_type="WorkerExit", error_message="exit code 9")

        accepted = registry.record({"job_id": job.job_id, "status": "ok", "best_metric": 1.0})

        assert accepted is False
        (outcome,) = registry.outcomes()
        assert outcome.status == "failed"
        assert outcome.error_type == "WorkerExit"


class TestParallelPath:
    @pytest.mark.parametrize(
        "runner",
        [run_exit_before_publish, run_sigkill_before_publish],
        ids=["os_exit", "sigkill"],
    )
    def test_should_fail_the_job_whose_worker_died_before_publishing(self, tmp_path, runner):
        doomed = _job(tmp_path, "doomed", die=True)
        survivors = [_job(tmp_path, f"ok-{i}") for i in range(2)]
        jobs = [doomed, *survivors]
        results = _pool(tmp_path, _dispatch(runner)).run(jobs)

        by_id = _by_id(results)
        assert sorted(by_id) == sorted(j.job_id for j in jobs)
        assert by_id[doomed.job_id]["outcome"] == "failed"
        assert by_id[doomed.job_id]["error_type"] == "WorkerExit"
        assert "exited with code" in by_id[doomed.job_id]["error"]
        assert attempts_of(doomed) == 1
        for job in survivors:
            assert by_id[job.job_id]["status"] == "ok"

    def test_should_retry_oom_in_the_pool_path_and_succeed(self, tmp_path):
        job = _job(tmp_path, "pool-oom-once", oom_times=1)

        results = _pool(tmp_path, run_oom_then_succeed).run([job, _job(tmp_path, "ok")])

        assert attempts_of(job) == 2
        assert _by_id(results)[job.job_id]["status"] == "ok"
        assert _by_id(results)[job.job_id]["attempts"] == 2

    def test_should_exhaust_oom_retries_with_the_same_count_as_the_sequential_path(self, tmp_path):
        job = _job(tmp_path, "pool-oom-always")

        results = _pool(tmp_path, run_always_oom).run([job, _job(tmp_path, "ok")])

        assert attempts_of(job) == MAX_OOM_RETRIES + 1
        result = _by_id(results)[job.job_id]
        assert result["status"] == "oom"
        assert result["outcome"] == "failed"
        assert result["attempts"] == MAX_OOM_RETRIES + 1


def run_dispatch_exit(job: TrainingJob) -> float:
    return run_exit_before_publish(job) if job.hyperparams.get("die") else run_succeed(job)


def run_dispatch_sigkill(job: TrainingJob) -> float:
    return run_sigkill_before_publish(job) if job.hyperparams.get("die") else run_succeed(job)


def _dispatch(runner):
    """Runner that kills only the job flagged ``die`` and succeeds otherwise."""
    return run_dispatch_exit if runner is run_exit_before_publish else run_dispatch_sigkill
