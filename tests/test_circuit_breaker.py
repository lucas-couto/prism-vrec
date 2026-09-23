"""A repeating non-OOM fault must stop the queue, not consume it.

A poisoned CUDA context (Xid 8 launch timeout, wedged driver) fails
every job it touches within seconds.  Each failure was already isolated
per job, so the orchestrator kept pulling: one hardware fault spent the
whole remaining queue against itself and reported thousands of "failed"
cells that never really ran.  The breaker stops after
``MAX_CONSECUTIVE_ERRORS`` and CANCELS the rest, so the outcome ledger
keeps "never attempted" distinct from "attempted and failed".

The runner is injected through ``job_runner`` so the real accounting,
retry and cancellation code runs on CPU without a dataset;
``job.processed_dir`` doubles as the cross-process attempt counter.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.parallel import (
    MAX_CONSECUTIVE_ERRORS,
    MAX_OOM_RETRIES,
    OUTCOME_CANCELLED,
    TrainingJob,
    _updated_error_streak,
)
from tests.test_parallel_job_outcomes import (
    _by_id,
    _count_attempt,
    _job,
    _sequential,
    attempts_of,
    run_always_oom,
    run_non_oom_error,
    run_succeed,
)


def run_error_unless_flagged(job: TrainingJob) -> float:
    """Error unless the job carries ``ok=True``.

    Lets a test lay out an exact error/success pattern along the queue,
    which is what decides whether a streak ever reaches the threshold.
    """
    if job.hyperparams.get("ok"):
        return run_succeed(job)
    return run_non_oom_error(job)


class TestConsecutiveErrorStreak:
    def test_should_clear_the_streak_on_a_success(self) -> None:
        assert _updated_error_streak(4, "ok") == 0

    def test_should_extend_the_streak_on_an_error(self) -> None:
        assert _updated_error_streak(4, "error") == 5

    def test_should_leave_the_streak_untouched_on_an_oom(self) -> None:
        assert _updated_error_streak(4, "oom") == 4


class TestBreakerTrips:
    def test_should_cancel_the_rest_of_the_queue_after_the_threshold(self, tmp_path: Path) -> None:
        n_jobs = MAX_CONSECUTIVE_ERRORS + 7
        jobs = [_job(tmp_path, f"j{i}") for i in range(n_jobs)]

        results = _sequential(tmp_path, run_non_oom_error).run(jobs)

        by_id = _by_id(results)
        statuses = [by_id[job.job_id]["status"] for job in jobs]
        assert statuses[:MAX_CONSECUTIVE_ERRORS] == ["error"] * MAX_CONSECUTIVE_ERRORS
        assert set(statuses[MAX_CONSECUTIVE_ERRORS:]) == {OUTCOME_CANCELLED}

    def test_should_not_attempt_a_job_it_cancelled(self, tmp_path: Path) -> None:
        jobs = [_job(tmp_path, f"j{i}") for i in range(MAX_CONSECUTIVE_ERRORS + 7)]

        _sequential(tmp_path, run_non_oom_error).run(jobs)

        attempted = [job for job in jobs if attempts_of(job) > 0]
        assert len(attempted) == MAX_CONSECUTIVE_ERRORS

    def test_should_give_every_submitted_job_a_terminal_outcome(self, tmp_path: Path) -> None:
        """A cancelled job is still accounted for: the step reconciles
        results against submitted ids, and a job with no outcome at all
        would be reported as ``unaccounted`` instead of cancelled."""
        jobs = [_job(tmp_path, f"j{i}") for i in range(MAX_CONSECUTIVE_ERRORS + 7)]

        results = _sequential(tmp_path, run_non_oom_error).run(jobs)

        assert set(_by_id(results)) == {job.job_id for job in jobs}

    def test_should_keep_the_cancelled_jobs_out_of_the_success_count(self, tmp_path: Path) -> None:
        jobs = [_job(tmp_path, f"j{i}") for i in range(MAX_CONSECUTIVE_ERRORS + 7)]

        results = _sequential(tmp_path, run_non_oom_error).run(jobs)

        assert not [r for r in results if r.get("status") == "ok"]


class TestBreakerHoldsFire:
    def test_should_not_trip_when_a_success_breaks_the_streak(self, tmp_path: Path) -> None:
        """Every fourth job succeeds, so the streak never reaches the
        threshold and the whole queue is still attempted."""
        jobs = [_job(tmp_path, f"j{i}", ok=(i % 4 == 3)) for i in range(4 * MAX_CONSECUTIVE_ERRORS)]

        results = _sequential(tmp_path, run_error_unless_flagged).run(jobs)

        statuses = {r["status"] for r in results}
        assert OUTCOME_CANCELLED not in statuses
        assert all(attempts_of(job) > 0 for job in jobs)

    def test_should_not_trip_on_repeated_oom(self, tmp_path: Path) -> None:
        """An OOM has its own retry-and-escalate path; a queue of
        irreducible OOMs must exhaust it, not be cancelled by the
        breaker that exists for poisoned contexts."""
        jobs = [_job(tmp_path, f"j{i}") for i in range(MAX_CONSECUTIVE_ERRORS + 3)]

        results = _sequential(tmp_path, run_always_oom).run(jobs)

        by_id = _by_id(results)
        assert not [r for r in results if r["status"] == OUTCOME_CANCELLED]
        for job in jobs:
            assert by_id[job.job_id]["error_type"] == "OutOfMemoryError"
            assert attempts_of(job) == MAX_OOM_RETRIES + 1


def test_should_escalate_to_lazy_before_the_breaker_sees_an_error(
    tmp_path: Path,
) -> None:
    """The OOM path escalates to lazy reads and can then succeed; that
    success must clear whatever streak preceded it."""

    def runner(job: TrainingJob) -> float:
        _count_attempt(job)
        if not job.lazy_features:
            raise __import__("torch").cuda.OutOfMemoryError("injected")
        return 0.75

    jobs = [_job(tmp_path, f"j{i}") for i in range(MAX_CONSECUTIVE_ERRORS + 2)]

    results = _sequential(tmp_path, runner).run(jobs)

    assert {r["status"] for r in results} == {"ok"}
