"""Progress, ETA and a per-cell breakdown for one queue of training jobs.

The orchestrator used to log progress only on the parallel path, so a
run with ``resources.workers.training: 1`` -- the pinned value -- went
from the first job to the last without a single progress or ETA line:
13 068 jobs followed by counting ``Starting:`` lines by eye.  This
module is the shared accounting both paths feed.

The ETA deliberately does NOT divide total elapsed by completed jobs.
Cost per job varies by an order of magnitude across the grid (a BPR
cell on amazon_men against an ACF cell on amazon_women), so a global
mean is confidently wrong for whatever is left.  Each ``(model,
dataset)`` cell is projected from its OWN observed mean, and only a
cell with no finished job yet falls back to the global mean.  With no
sample at all there is no estimate, and none is invented.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from src.utils.tables import format_duration

#: Seconds between progress lines.  Matches the parallel path's cadence.
PROGRESS_LOG_S = 30.0


@dataclass
class CellProgress:
    """One ``(model, dataset)`` cell's share of the queue."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    seconds: float = 0.0

    @property
    def finished(self) -> int:
        return self.succeeded + self.failed

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.finished)

    @property
    def mean_seconds(self) -> float | None:
        """Seconds observed per SUCCESS, or None without one.

        Failures are not a sample of how long the work takes: a cell
        that crashed three times in 8 seconds each would otherwise be
        projected at 8 s per job, and 594 pending ACF jobs would be
        advertised as an hour of work instead of days.  The numerator
        still carries every attempt, failures and OOM retries included,
        so a cell that wastes time on failures is projected as costing
        that time -- what changes is that a cell with no success yet has
        no mean of its own at all.
        """
        return self.seconds / self.succeeded if self.succeeded else None


@dataclass
class JobProgress:
    """Live counters for a submitted queue, keyed by ``(model, dataset)``."""

    cells: dict[tuple[str, str], CellProgress] = field(default_factory=dict)
    total: int = 0
    clock: Callable[[], float] = time.time
    log_every_s: float = PROGRESS_LOG_S
    _started: float = field(default=0.0, init=False)
    _last_log: float = field(default=0.0, init=False)

    @classmethod
    def of_jobs(cls, jobs: Iterable, **kwargs) -> JobProgress:
        """Build the counters from the submitted jobs, before any runs."""
        progress = cls(**kwargs)
        for job in jobs:
            cell = progress.cells.setdefault((job.model_name, job.dataset_name), CellProgress())
            cell.total += 1
            progress.total += 1
        progress._started = progress.clock()
        progress._last_log = progress._started
        return progress

    # -- accounting ------------------------------------------------------

    def finish(self, job, status: str, duration: float, *, terminal: bool = True) -> None:
        """Record one finished ATTEMPT of a job.

        Every attempt is real time spent and always lands in the cell's
        seconds, so a cell that burns retries is projected as costing
        that time.  Only a *terminal* attempt moves the done counters:
        an OOM that is owed a retry would otherwise be counted now and
        again when it really ends, and the queue would report more
        finished jobs than it was given.
        """
        cell = self.cells.setdefault((job.model_name, job.dataset_name), CellProgress())
        cell.seconds += max(0.0, duration)
        if not terminal:
            return
        if status == "ok":
            cell.succeeded += 1
        else:
            cell.failed += 1

    @property
    def finished(self) -> int:
        return sum(cell.finished for cell in self.cells.values())

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.finished)

    @property
    def elapsed(self) -> float:
        return self.clock() - self._started

    # -- projection ------------------------------------------------------

    def _global_mean(self) -> float | None:
        """Seconds per success across every cell, or None without one."""
        succeeded = sum(cell.succeeded for cell in self.cells.values())
        if not succeeded:
            return None
        return sum(cell.seconds for cell in self.cells.values()) / succeeded

    def eta_seconds(self) -> float | None:
        """Remaining wall-clock, or None when nothing has finished yet.

        Sequential execution is assumed: the per-cell projections add up.
        A pool of N workers divides this, which the caller knows and
        this module deliberately does not.
        """
        fallback = self._global_mean()
        if fallback is None:
            return None
        return sum(
            (cell.mean_seconds if cell.mean_seconds is not None else fallback) * cell.remaining
            for cell in self.cells.values()
        )

    # -- rendering -------------------------------------------------------

    def due(self) -> bool:
        """True at most once per ``log_every_s``; resets the window."""
        now = self.clock()
        if now - self._last_log < self.log_every_s:
            return False
        self._last_log = now
        return True

    def cell_mean(self, key: tuple[str, str]) -> tuple[float | None, bool]:
        """``(seconds per job, is_a_fallback)`` for one cell.

        Lets the renderer show the same number the total is built from,
        flagged as borrowed, instead of a bare dash next to an ETA that
        silently charged the cell anyway.
        """
        cell = self.cells[key]
        if cell.mean_seconds is not None:
            return cell.mean_seconds, False
        return self._global_mean(), True

    def line(self, *, workers: int = 1) -> str:
        """One-line progress summary with the projected remainder."""
        finished = self.finished
        pct = 100 * finished / self.total if self.total else 0.0
        eta = self.eta_seconds()
        if eta is None:
            eta_text = "no estimate yet"
        else:
            eta_text = f"~{format_duration(eta / max(1, workers))} left"
        mean = self._global_mean()
        rate = f"{mean:.0f}s/job" if mean is not None else "-"
        return (
            f"Progress: {finished}/{self.total} ({pct:.1f}%) | {rate} | "
            f"elapsed {format_duration(self.elapsed)} | {eta_text}"
        )
