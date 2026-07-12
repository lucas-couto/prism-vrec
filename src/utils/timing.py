"""Pipeline timing accumulator.

Captures wall-clock durations at two granularities:

* **Per pipeline step**, a flat list of
  ``{name, started_at, duration_seconds}`` embedded in the run
  manifest under the ``steps`` key.  Always recorded by ``main.py``.
* **Per cell**, opt-in finer-grained log written to
  ``results/runs/<run_id>/step_timings.json``.  Hot loops in the
  expensive steps (extract, finetune, train, ...) wrap each cell
  with the :func:`time_cell` context manager so a researcher can
  audit how long every ``(dataset, extractor)`` or
  ``(dataset, embedding, recommender)`` combination took.

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

from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class _TimingRecorder:
    """Process-wide accumulator (thread-safe; not subprocess-safe)."""

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._cells: list[dict[str, Any]] = []
        self._run_dir: Path | None = None
        self._lock = Lock()

    def bind(self, run_dir: Path | str) -> None:
        with self._lock:
            self._run_dir = Path(run_dir)

    def record_step(self, name: str, started_at: str, duration_seconds: float) -> None:
        with self._lock:
            self._steps.append(
                {
                    "name": name,
                    "started_at": started_at,
                    "duration_seconds": round(duration_seconds, 3),
                }
            )

    def steps(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._steps)

    def cells(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._cells)

    def reset(self) -> None:
        """Clear all accumulated state.  Test-only escape hatch."""
        with self._lock:
            self._steps.clear()
            self._cells.clear()
            self._run_dir = None

    @contextmanager
    def time_cell(self, step: str, **labels: Any) -> Iterator[None]:
        started_at = _now_iso()
        start_perf = time.perf_counter()
        try:
            yield
        finally:
            duration = round(time.perf_counter() - start_perf, 3)
            with self._lock:
                self._cells.append(
                    {
                        "step": step,
                        "started_at": started_at,
                        "duration_seconds": duration,
                        "labels": labels,
                    }
                )
                self._flush_unsafe()

    def _flush_unsafe(self) -> None:
        """Persist the cell list to disk; caller already holds the lock."""
        if self._run_dir is None:
            return
        path = self._run_dir / "step_timings.json"
        payload = json.dumps(self._cells, indent=2)
        try:
            atomic_write(lambda tmp: Path(tmp).write_text(payload), path)
        except OSError as exc:
            logger.warning("failed to write %s: %r", path, exc)


_RECORDER = _TimingRecorder()


def bind_run_dir(run_dir: Path | str) -> None:
    """Bind the global recorder to a run directory.

    Called once by :func:`main.main` right after :func:`start_run`.
    The path is where :func:`time_cell` writes ``step_timings.json``.
    Until bound, per-cell timings are still accumulated in memory but
    not persisted.
    """
    _RECORDER.bind(run_dir)


def record_step(name: str, started_at: str, duration_seconds: float) -> None:
    """Append a top-level step timing (one entry per ``_run_step`` call)."""
    _RECORDER.record_step(name, started_at, duration_seconds)


def time_cell(step: str, **labels: Any):
    """Context manager that times one cell of work.

    Usage::

        with time_cell("extract", dataset=name, extractor=ext, dim=d):
            do_extraction()

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
