"""The run must say what it is doing and how much is left.

Sequential execution reported nothing between the first job and the
last: ``_maybe_log_progress`` was reachable only from the pool path, so
a 13 068-job grid on one worker logged no progress and no ETA at all.
These tests pin the accounting (per-cell means, the projection, the
streak-free counters) and the rendering, all of which are pure and run
without a GPU or a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.job_progress import CellProgress, JobProgress
from src.utils.progress import render_cells, render_plan, render_timeline, step_scale
from src.utils.tables import format_duration, render_table


@dataclass
class FakeJob:
    """Only the two fields the tracker keys on."""

    model_name: str
    dataset_name: str


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _jobs(*pairs: tuple[str, str, int]) -> list[FakeJob]:
    jobs: list[FakeJob] = []
    for model, dataset, count in pairs:
        jobs.extend(FakeJob(model, dataset) for _ in range(count))
    return jobs


class TestFormatDuration:
    def test_should_report_seconds_below_a_minute(self) -> None:
        assert format_duration(45) == "45s"

    def test_should_report_minutes_and_seconds_below_an_hour(self) -> None:
        assert format_duration(125) == "2m 05s"

    def test_should_report_hours_and_minutes_below_a_day(self) -> None:
        assert format_duration(3 * 3600 + 12 * 60) == "3h 12m"

    def test_should_report_days_and_hours_above_a_day(self) -> None:
        assert format_duration(11 * 86400 + 2 * 3600) == "11d 02h"

    def test_should_not_render_a_negative_duration_as_a_time(self) -> None:
        assert format_duration(-1) == "-"


class TestCounters:
    def test_should_total_every_submitted_job(self) -> None:
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 3), ("vbpr", "men", 2)))

        assert progress.total == 5
        assert progress.remaining == 5
        assert progress.finished == 0

    def test_should_group_the_queue_by_recommender_and_dataset(self) -> None:
        progress = JobProgress.of_jobs(
            _jobs(("bpr", "men", 3), ("bpr", "women", 1), ("vbpr", "men", 2))
        )

        assert progress.cells[("bpr", "men")].total == 3
        assert progress.cells[("bpr", "women")].total == 1
        assert progress.cells[("vbpr", "men")].total == 2

    def test_should_separate_successes_from_failures(self) -> None:
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 2)))

        progress.finish(FakeJob("bpr", "men"), "ok", 10.0)
        progress.finish(FakeJob("bpr", "men"), "error", 2.0)

        cell = progress.cells[("bpr", "men")]
        assert (cell.succeeded, cell.failed, cell.finished) == (1, 1, 2)
        assert progress.remaining == 0


class TestProjection:
    def test_should_give_no_estimate_before_anything_finishes(self) -> None:
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 4)))

        assert progress.eta_seconds() is None
        assert "no estimate yet" in progress.line()

    def test_should_project_a_cell_from_its_own_mean(self) -> None:
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 4)))

        progress.finish(FakeJob("bpr", "men"), "ok", 10.0)

        assert progress.eta_seconds() == 30.0

    def test_should_not_charge_a_slow_cell_at_a_fast_cell_rate(self) -> None:
        """The whole point of the per-cell projection: one BPR sample
        must not make 10 pending ACF jobs look 10s each."""
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 2), ("acf", "women", 11)))

        progress.finish(FakeJob("bpr", "men"), "ok", 10.0)
        progress.finish(FakeJob("acf", "women"), "ok", 100.0)

        # 1 BPR left at 10s + 10 ACF left at 100s.
        assert progress.eta_seconds() == 10.0 + 1000.0

    def test_should_fall_back_to_the_global_mean_for_an_unsampled_cell(self) -> None:
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 1), ("acf", "women", 2)))

        progress.finish(FakeJob("bpr", "men"), "ok", 12.0)

        # No ACF sample yet: its 2 jobs are charged the global mean.
        assert progress.eta_seconds() == 24.0

    def test_should_divide_the_projection_across_pool_workers(self) -> None:
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 5)))
        progress.finish(FakeJob("bpr", "men"), "ok", 100.0)

        assert "~1m 40s left" in progress.line(workers=4)

    def test_should_charge_a_retry_to_the_success_it_paid_for(self) -> None:
        """Both attempts are real time; the denominator is the success,
        so the cell is projected at what a success actually costs."""
        progress = JobProgress.of_jobs(_jobs(("acf", "women", 2)))

        progress.finish(FakeJob("acf", "women"), "oom", 30.0, terminal=False)
        progress.finish(FakeJob("acf", "women"), "ok", 60.0)

        assert progress.cells[("acf", "women")].mean_seconds == 90.0

    def test_should_not_count_a_retried_attempt_as_a_finished_job(self) -> None:
        """An OOM owed a retry is time spent, not an outcome: counting
        it here and again when it really ends would report more finished
        jobs than the queue was given."""
        progress = JobProgress.of_jobs(_jobs(("acf", "women", 1)))

        progress.finish(FakeJob("acf", "women"), "oom", 30.0, terminal=False)

        assert progress.finished == 0
        assert progress.remaining == 1

        progress.finish(FakeJob("acf", "women"), "ok", 60.0)

        assert progress.finished == 1
        assert progress.remaining == 0

    def test_should_not_project_a_cell_from_its_failures_alone(self) -> None:
        """Three fast crashes must not advertise 594 pending jobs as an
        hour of work: a cell with no success has no mean of its own."""
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 2), ("acf", "women", 100)))
        progress.finish(FakeJob("bpr", "men"), "ok", 60.0)

        for _ in range(3):
            progress.finish(FakeJob("acf", "women"), "error", 8.0)

        acf = progress.cells[("acf", "women")]
        assert acf.mean_seconds is None
        mean, borrowed = progress.cell_mean(("acf", "women"))
        assert borrowed is True
        assert mean is not None and mean > 8.0


class TestCadence:
    def test_should_not_log_twice_inside_one_window(self) -> None:
        clock = FakeClock()
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 2)), clock=clock, log_every_s=30.0)

        clock.advance(31)
        assert progress.due() is True
        clock.advance(5)
        assert progress.due() is False

    def test_should_log_again_after_the_window(self) -> None:
        clock = FakeClock()
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 2)), clock=clock, log_every_s=30.0)

        clock.advance(31)
        progress.due()
        clock.advance(31)
        assert progress.due() is True


class TestTableRendering:
    def test_should_pad_every_column_to_its_widest_entry(self) -> None:
        lines = render_table(["a", "bbbb"], [["cccccc", "d"]])

        assert lines[0] == "a      | bbbb"
        assert lines[1] == "------ | ----"
        assert lines[2] == "cccccc | d"

    def test_should_render_one_row_per_cell(self) -> None:
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 1), ("vbpr", "women", 1)))

        lines = render_cells(progress)

        assert any("bpr" in line and "men" in line for line in lines)
        assert any("vbpr" in line and "women" in line for line in lines)

    def test_should_mark_an_unsampled_cell_rather_than_guess_it(self) -> None:
        progress = JobProgress.of_jobs(_jobs(("acf", "women", 3)))

        body = "\n".join(render_cells(progress))

        assert "0/3" in body
        assert "-" in body

    def test_should_flag_a_borrowed_mean_instead_of_hiding_it(self) -> None:
        """The total charges the unsampled cell the global mean, so the
        row shows that number marked, not a dash."""
        progress = JobProgress.of_jobs(_jobs(("bpr", "men", 1), ("acf", "women", 4)))
        progress.finish(FakeJob("bpr", "men"), "ok", 30.0)

        rows = [line for line in render_cells(progress) if line.startswith("acf")]

        assert rows and "~" in rows[0]


class TestPlanTable:
    def _config(self) -> dict:
        return {
            "datasets": ["men", "women"],
            "extractors_enabled": ["resnet50", "vit_b16", "clip_vitb32"],
            "recommenders_enabled": ["bpr", "vbpr", "acf"],
            "fusion_strategies_enabled": ["concat", "sum"],
            "folds": {"enabled": True, "k": 2},
        }

    def test_should_number_every_step_in_order(self) -> None:
        lines = render_plan(
            ["download", "train"], self._config(), condition="frozen", run_both=False
        )

        body = "\n".join(lines)
        assert "1 | download" in body
        assert "2 | train" in body

    def test_should_report_the_scale_it_can_read_off_the_config(self) -> None:
        config = self._config()

        assert step_scale("download", config) == "2 datasets"
        assert step_scale("extract", config) == "2 datasets x 3 extractors"
        assert step_scale("fuse", config) == "2 datasets x 2 strategies"
        assert step_scale("folds", config) == "2 folds"

    def test_should_say_folds_is_disabled_when_the_block_is_off(self) -> None:
        assert step_scale("folds", {"folds": {"enabled": False}}) == "disabled"

    def test_should_mark_the_steps_that_run_once_per_condition(self) -> None:
        lines = render_plan(
            ["train"],
            self._config(),
            condition="both",
            run_both=True,
            condition_steps=["train"],
        )

        assert any("runs twice" in line for line in lines)

    def test_should_not_mark_condition_steps_on_a_single_condition_run(self) -> None:
        lines = render_plan(
            ["train"],
            self._config(),
            condition="frozen",
            run_both=False,
            condition_steps=["train"],
        )

        assert not any("runs twice" in line for line in lines)


class TestTimeline:
    """The records come from the real recorder, never from a literal.

    The first version of this renderer read ``step`` / ``duration_s``,
    which :func:`record_step` has never written (its keys are ``name``
    and ``duration_seconds``): every row rendered as ``pending`` next to
    a title claiming one step was complete. Literal fixtures encoded the
    same wrong guess and passed, so these build their input through the
    producer and cannot drift from it again.
    """

    @staticmethod
    def _timings(*steps: tuple[str, float, bool]) -> list[dict]:
        from src.utils.timing import record_step, reset_for_tests, step_timings

        reset_for_tests()
        for name, duration, skipped in steps:
            record_step(name, "2026-09-12T00:00:00Z", duration, None, skipped)
        return step_timings()

    def test_should_report_a_recorded_step_as_done_with_its_elapsed(self) -> None:
        timings = self._timings(("download", 18.3, False))

        body = "\n".join(render_timeline(["download", "train"], timings))

        assert "download" in body and "done" in body and "18s" in body

    def test_should_not_leave_a_recorded_step_showing_as_pending(self) -> None:
        """The defect the literal fixtures hid: a counted record whose
        row never matched, so the table said 1/2 complete with both
        rows pending."""
        timings = self._timings(("download", 18.3, False))

        rows = [
            line
            for line in render_timeline(["download", "train"], timings)
            if line.startswith("download") or " download" in line
        ]

        assert rows
        assert not any("pending" in row for row in rows)

    def test_should_match_a_condition_labelled_record_to_its_step(self) -> None:
        """``_run_step`` labels a condition step ``train (frozen)``."""
        timings = self._timings(("train (frozen)", 60.0, False))

        body = "\n".join(render_timeline(["train"], timings))

        assert "pending" not in body
        assert "1m 00s" in body

    def test_should_report_a_skipped_step_as_skipped_not_done(self) -> None:
        timings = self._timings(("extract", 31.6, True))

        body = "\n".join(render_timeline(["extract"], timings))

        assert "skipped" in body
        assert "done" not in body

    def test_should_mark_the_running_step_apart_from_the_pending_ones(self) -> None:
        body = "\n".join(render_timeline(["download", "train"], [], current="download"))

        assert "running" in body
        assert "pending" in body

    def test_should_count_only_the_recorded_steps_as_complete(self) -> None:
        timings = self._timings(("download", 10.0, False))

        title = render_timeline(["download", "train", "evaluate"], timings)[0]

        assert "1/3 complete" in title

    def test_should_total_the_elapsed_of_the_recorded_steps(self) -> None:
        timings = self._timings(("download", 18.0, False), ("preprocess", 42.0, False))

        title = render_timeline(["download", "preprocess"], timings)[0]

        assert "1m 00s of step time" in title


def test_cell_progress_has_no_mean_without_a_sample() -> None:
    assert CellProgress(total=3).mean_seconds is None


class TestOrchestratorAccounting:
    """The counters must survive the real retry path, not just the unit
    arithmetic: an OOM-then-success job reports twice and must still be
    one finished job out of one submitted."""

    def test_should_count_a_retried_job_once_through_the_orchestrator(self, tmp_path) -> None:
        from tests.test_parallel_job_outcomes import _job, _sequential, run_oom_then_succeed

        jobs = [_job(tmp_path, f"j{i}", oom_times=1) for i in range(3)]

        orchestrator = _sequential(tmp_path, run_oom_then_succeed)
        results = orchestrator.run(jobs)

        progress = orchestrator._progress
        assert [r["status"] for r in results] == ["ok"] * 3
        assert progress.total == 3
        assert progress.finished == 3
        assert progress.remaining == 0

    def test_should_charge_the_retry_time_to_the_cell(self, tmp_path) -> None:
        from tests.test_parallel_job_outcomes import _job, _sequential, run_oom_then_succeed

        jobs = [_job(tmp_path, "j0", oom_times=1)]

        orchestrator = _sequential(tmp_path, run_oom_then_succeed)
        orchestrator.run(jobs)

        cell = orchestrator._progress.cells[("vbpr", "synthetic")]
        assert cell.succeeded == 1
        assert cell.failed == 0


def test_scale_reads_keys_the_merged_config_really_has() -> None:
    """Pins the contract between the plan table and the loader.

    The first version counted `fusions_enabled`, which the merged YAML
    has never contained (the key is `fusion_strategies_enabled`), so the
    plan advertised "4 datasets x 0 strategies" for a fuse step that
    went on to build 96 fusions. A literal fixture cannot catch that.
    """
    from src.utils.config import load_config

    config = load_config()

    for key in (
        "datasets",
        "extractors_enabled",
        "recommenders_enabled",
        "fusion_strategies_enabled",
    ):
        value = config.get(key)
        assert isinstance(value, list) and value, f"{key} missing from the merged config"

    for step in ("download", "extract", "fuse", "train"):
        assert " 0 " not in f" {step_scale(step, config)} "
