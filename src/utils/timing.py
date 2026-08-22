"""Pipeline timing accumulator.

Captures wall-clock durations at two granularities:

* **Per pipeline step**, a flat list of
  ``{name, started_at, duration_seconds, telemetry}`` embedded in the
  run manifest under the ``steps`` key.  Always recorded by
  ``main.py``.  The ``telemetry`` block carries throughput and cost
  aggregates for the step's window; see :mod:`src.utils.telemetry`.
  The manifest only receives this list when the run closes, so the
  same entries are also flushed after every step into the sidecar
  ``results/runs/<run_id>/steps.json``.  That is what a researcher
  reads while the pipeline is still running, and what survives a run
  that is killed before ``finish_run`` — steps with no cells at all
  (``preprocess``, ``report``) have no other trace.
* **Per cell**, opt-in finer-grained log written to
  ``results/runs/<run_id>/step_timings.json``.  Hot loops in the
  expensive steps (download, extract, finetune, ...) wrap each cell
  with the :func:`time_cell` context manager so a researcher can
  audit how long every ``(dataset, extractor)`` or
  ``(dataset, embedding, recommender)`` combination took.

A cell that found its output already on disk did no work, so timing
and costing it is meaningless: it would report a fraction of a second
and zero energy for an extraction that actually cost an hour on an
earlier run.  Such a cell calls :meth:`Cell.skip` on the handle
``time_cell`` yields and leaves no entry behind.  A step whose cells
were *all* skipped is recorded as ``skipped`` with no telemetry block,
so re-running a finished pipeline no longer dilutes the manifest with
no-op windows.

Both levels accumulate in a module-level singleton so a step
deeply nested in a loop never has to thread a recorder through
every function signature.  The recorder is thread-safe (multiple
threads within one process append concurrently), but it is **not**
subprocess-safe, a worker spawned via :mod:`multiprocessing` or
joblib runs in its own process and has its own (empty) singleton.
Per-cell timings for parallel hyperparameter search are therefore
deliberately omitted; Optuna's own study database covers that
breakdown.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from src.utils import telemetry
from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Cell:
    """Handle yielded by :func:`time_cell` so a cell can describe itself.

    A step that discovers its output already exists calls :meth:`skip`;
    the surrounding context manager then records nothing at all.  A step
    that only learns part of a cell's identity *while doing the work*
    (a download that weighs its dataset once the bytes have landed)
    calls :meth:`label` to add it before the entry is written.
    """

    __slots__ = ("skipped", "reason", "extra_labels")

    def __init__(self) -> None:
        self.skipped = False
        self.reason: str | None = None
        self.extra_labels: dict[str, Any] = {}

    def skip(self, reason: str | None = None) -> None:
        """Mark this cell as "no work done" so it is never recorded.

        :param reason:
            Optional human-readable note (e.g. ``"embeddings exist"``).
            Kept for the caller's own logging; it is not persisted,
            because a skipped cell produces no entry.
        """
        self.skipped = True
        self.reason = reason

    def label(self, **labels: Any) -> None:
        """Add labels that are only known once the work has run.

        Merged over the labels passed to :func:`time_cell`, so a late
        value wins over a placeholder of the same name.  Ignored when
        the cell is skipped, since a skipped cell records nothing.

        :param labels: Arbitrary key/value pairs for the entry's
            ``labels`` dict (e.g. ``size_mb=812.4``).
        """
        self.extra_labels.update(labels)


class _TimingRecorder:
    """Process-wide accumulator (thread-safe; not subprocess-safe)."""

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._cells: list[dict[str, Any]] = []
        self._skipped_cells = 0
        self._run_dir: Path | None = None
        self._lock = Lock()

    def bind(self, run_dir: Path | str) -> None:
        with self._lock:
            self._run_dir = Path(run_dir)
            # Anything recorded before the bind (nothing today, but the
            # order in ``main`` is not this module's to guarantee) is
            # persisted now rather than waiting for the next entry.
            self._flush_steps_unsafe()
            self._flush_cells_unsafe()

    def record_step(
        self,
        name: str,
        started_at: str,
        duration_seconds: float,
        metrics: dict[str, Any] | None = None,
        skipped: bool = False,
    ) -> None:
        entry: dict[str, Any] = {
            "name": name,
            "started_at": started_at,
            "duration_seconds": round(duration_seconds, 3),
        }
        if skipped:
            # No new work: the window measured only the existence checks,
            # so its throughput and cost figures describe nothing.
            entry["skipped"] = True
        elif metrics:
            entry["telemetry"] = metrics
        with self._lock:
            self._steps.append(entry)
            self._flush_steps_unsafe()

    def steps(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._steps)

    def cells(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._cells)

    def cell_counts(self) -> tuple[int, int]:
        """Return ``(recorded, skipped)`` cell counts so far."""
        with self._lock:
            return len(self._cells), self._skipped_cells

    def note_skipped_cell(self) -> None:
        with self._lock:
            self._skipped_cells += 1

    def reset(self) -> None:
        """Clear all accumulated state.  Test-only escape hatch."""
        with self._lock:
            self._steps.clear()
            self._cells.clear()
            self._skipped_cells = 0
            self._run_dir = None

    @contextmanager
    def time_cell(self, step: str, **labels: Any) -> Iterator[Cell]:
        cell = Cell()
        started_at = _now_iso()
        start_perf = time.perf_counter()
        marker = telemetry.mark()
        try:
            yield cell
        finally:
            if cell.skipped:
                # Nothing ran: counted, not recorded.  The count is what
                # lets ``_run_step`` tell "step did no new work" from
                # "step has no cells at all" (preprocess, report).
                with self._lock:
                    self._skipped_cells += 1
            else:
                duration = round(time.perf_counter() - start_perf, 3)
                entry: dict[str, Any] = {
                    "step": step,
                    "started_at": started_at,
                    "duration_seconds": duration,
                    "labels": {**labels, **cell.extra_labels},
                }
                # The cell slices the same run-wide sample series the
                # enclosing step will slice, so nesting costs nothing extra.
                metrics = telemetry.summarise_since(marker)
                if metrics:
                    entry["telemetry"] = metrics
                with self._lock:
                    self._cells.append(entry)
                    self._flush_cells_unsafe()

    def _flush_cells_unsafe(self) -> None:
        """Persist the cell list to disk; caller already holds the lock."""
        self._write_unsafe("step_timings.json", self._cells)

    def _flush_steps_unsafe(self) -> None:
        """Persist the step list to disk; caller already holds the lock."""
        self._write_unsafe("steps.json", self._steps)

    def _write_unsafe(self, filename: str, entries: list[dict[str, Any]]) -> None:
        """Atomically dump *entries* into ``<run_dir>/<filename>``.

        An empty list writes nothing: a run that recorded no cells
        should leave no sidecar behind rather than an empty array
        that reads as "measured, found nothing".
        """
        if self._run_dir is None or not entries:
            return
        path = self._run_dir / filename
        payload = json.dumps(entries, indent=2)
        try:
            atomic_write(lambda tmp: Path(tmp).write_text(payload), path)
        except OSError as exc:
            logger.warning("failed to write %s: %r", path, exc)


_RECORDER = _TimingRecorder()


def bind_run_dir(run_dir: Path | str) -> None:
    """Bind the global recorder to a run directory.

    Called once by :func:`main.main` right after :func:`start_run`.
    The path is where :func:`time_cell` writes ``step_timings.json``
    and :func:`record_step` writes ``steps.json``.  Until bound, both
    levels are still accumulated in memory but not persisted.
    """
    _RECORDER.bind(run_dir)


def record_step(
    name: str,
    started_at: str,
    duration_seconds: float,
    metrics: dict[str, Any] | None = None,
    skipped: bool = False,
) -> None:
    """Append a top-level step timing (one entry per ``_run_step`` call).

    ``metrics`` is the telemetry summary for the step's window, as
    returned by :func:`src.utils.telemetry.summarise_since`.  It is
    optional so callers that do not sample (tests, embedders) keep the
    three-argument form.

    ``skipped`` marks a step that found all of its work already done.
    Its entry carries ``skipped: true`` and no telemetry, because the
    measured window covers existence checks rather than computation.

    The updated list is flushed to ``<run_dir>/steps.json`` on every
    call, so an interrupted run still documents the steps it did run.
    """
    _RECORDER.record_step(name, started_at, duration_seconds, metrics, skipped)


def time_cell(step: str, **labels: Any):
    """Context manager that times one cell of work.

    Usage::

        with time_cell("extract", dataset=name, extractor=ext, dim=d):
            do_extraction()

    The manager yields a :class:`Cell`.  A cell that turns out to have
    nothing to do disowns itself, leaving no entry in
    ``step_timings.json``::

        with time_cell("extract", dataset=name, extractor=ext) as cell:
            if not _extract_for_config(...):
                cell.skip("embeddings exist")

    A label whose value only exists once the work has run is added on
    the handle instead, and lands in the same ``labels`` dict::

        with time_cell("download", dataset=name) as cell:
            provider.download()
            cell.label(size_mb=weigh(provider.raw_dir))

    *labels* are arbitrary keyword arguments that end up under
    ``labels`` in the JSON entry.  They make every line in
    ``step_timings.json`` self-describing: a downstream tool plotting
    "extract time per backbone" can group on ``labels.extractor``
    without inferring it from a position-encoded string.
    """
    return _RECORDER.time_cell(step, **labels)


def step_timings() -> list[dict[str, Any]]:
    """Return a copy of the recorded per-step timings."""
    return _RECORDER.steps()


def cell_timings() -> list[dict[str, Any]]:
    """Return a copy of the recorded per-cell timings."""
    return _RECORDER.cells()


def note_skipped_cell() -> None:
    """Count one cell that was skipped *before* a timer was ever started.

    Steps whose skip decision is available up front (``evaluate``,
    ``evaluate_finetuning``) short-circuit before entering
    :func:`time_cell`.  They call this instead so the step-level
    "did no new work" verdict sees them, without paying for a context
    manager that would record nothing.
    """
    _RECORDER.note_skipped_cell()


def cell_counts() -> tuple[int, int]:
    """Return ``(recorded, skipped)`` cell counts for the whole run.

    ``main._run_step`` snapshots this before and after each step to tell
    whether the step did any new work.
    """
    return _RECORDER.cell_counts()


def reset_for_tests() -> None:
    """Clear all accumulated state, for test isolation only."""
    _RECORDER.reset()


def now_iso() -> str:
    """UTC timestamp in ``YYYY-MM-DDTHH:MM:SSZ`` format.

    Public re-export of the helper used internally so callers in
    ``main.py`` can stamp ``started_at`` without importing
    ``datetime`` themselves.
    """
    return _now_iso()
