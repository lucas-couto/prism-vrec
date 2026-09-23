"""Escalation policy and audit trail of CUDA OOM recoveries.

Every retry the framework makes after a ``torch.cuda.OutOfMemoryError``
changes HOW a job runs, never WHAT it computes, so its result is
reused like any other.  That is exactly why each one must be traceable:
``results/runs/<run_id>/oom_recoveries.csv`` gets one row per event,
and the same event is logged at WARNING with the ``OOM recovery`` tag.

Escalation, per retry (:func:`escalate`):

1. ``lazy_features`` -- the feature matrix stops being resident and
   every gather is bounded and de-duplicated.  Skipped when the job
   already reads lazily (``resources.features.residency: lazy``).
2. ``micro_batches`` -- each BPR step is split into twice as many
   micro-batches with gradient accumulation.  The effective batch is
   unchanged and the accumulated gradient equals the full-batch one
   (:func:`src.utils.training.bpr_accumulated_step`); only dropout
   masks are drawn per micro-batch instead of per batch.

The ranking budget of the pool additionally halves on every retry
(``ranking_budget_factor``), as it did before this module existed.

=========================  ===============================================
column                     meaning
=========================  ===============================================
``recorded_at``            UTC timestamp of the event
``step``                   ``train`` (grid pool) or ``folds``
``dataset`` .. ``job``     identity of the job or fold cell
``hyperparams``            JSON of the job's hyperparameters
``attempt``                attempt that raised the OOM (or that succeeded)
``outcome``                ``retrying`` / ``recovered`` / ``failed``
``action``                 what the NEXT attempt changes (retrying only)
``lazy_features``          residency of the next (or final) attempt
``micro_batches``          micro-batches per step of that attempt
``ranking_budget_factor``  ranking budget multiplier of that attempt
``error``                  first line of the OOM message
=========================  ===============================================
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger
from src.utils.timing import now_iso

logger = get_logger(__name__)

ACTION_LAZY_FEATURES = "lazy_features"
ACTION_MICRO_BATCHES = "micro_batches"

OUTCOME_RETRYING = "retrying"
OUTCOME_RECOVERED = "recovered"
OUTCOME_FAILED = "failed"

#: Key under which the fold runner carries the escalation state inside
#: its private config copy (the ``execute`` callable keeps its shape).
RECOVERY_CONFIG_KEY = "_oom_recovery"

FILENAME = "oom_recoveries.csv"
COLUMNS = (
    "recorded_at",
    "step",
    "dataset",
    "model",
    "embedding",
    "job",
    "hyperparams",
    "attempt",
    "outcome",
    "action",
    "lazy_features",
    "micro_batches",
    "ranking_budget_factor",
    "error",
)


@dataclass(frozen=True)
class Escalation:
    """Execution settings of the next attempt and the change that led there."""

    lazy_features: bool
    micro_batches: int
    action: str


def escalate(*, lazy_features: bool, micro_batches: int) -> Escalation:
    """Return the settings of the attempt that follows an OOM.

    @param lazy_features - Whether the attempt that failed read features lazily.
    @param micro_batches - Micro-batches per step of the attempt that failed.
    @returns Lazy reads first; once lazy, twice the micro-batches.
    """
    if not lazy_features:
        return Escalation(True, max(1, micro_batches), ACTION_LAZY_FEATURES)
    return Escalation(True, max(1, micro_batches) * 2, ACTION_MICRO_BATCHES)


class _RecoveryRecorder:
    """Process-wide list of events, rewritten atomically on every record."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._run_dir: Path | None = None
        self._lock = Lock()

    def bind(self, run_dir: Path | str, *, resume: bool = False) -> None:
        with self._lock:
            self._run_dir = Path(run_dir)
            if resume:
                self._rows = _load_rows(self._run_dir / FILENAME) + self._rows
            self._flush_unsafe()

    def record(self, row: dict[str, Any]) -> None:
        full = {column: row.get(column, "") for column in COLUMNS}
        full["recorded_at"] = now_iso()
        full["error"] = str(full["error"]).strip().splitlines()[0][:300] if full["error"] else ""
        with self._lock:
            self._rows.append(full)
            self._flush_unsafe()
        logger.warning(
            "OOM recovery: step=%s job=%s attempt=%s outcome=%s action=%s "
            "lazy_features=%s micro_batches=%s ranking_budget_factor=%s",
            *(full[c] for c in ("step", "job", "attempt", "outcome", "action")),
            *(full[c] for c in ("lazy_features", "micro_batches", "ranking_budget_factor")),
        )

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._rows]

    def reset(self) -> None:
        with self._lock:
            self._rows = []
            self._run_dir = None

    def _flush_unsafe(self) -> None:
        if self._run_dir is None or not self._rows:
            return
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self._rows)
        path = self._run_dir / FILENAME
        try:
            atomic_write(lambda tmp: Path(tmp).write_text(buffer.getvalue()), path)
        except OSError as exc:
            logger.warning("failed to write %s: %r", path, exc)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Events an earlier attempt of the same run wrote to *path*."""
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return [
                {column: row.get(column, "") for column in COLUMNS} for row in csv.DictReader(fh)
            ]
    except OSError:
        return []


_RECORDER = _RecoveryRecorder()


def bind_run_dir(run_dir: Path | str, *, resume: bool = False) -> None:
    """Persist the events of this process into ``<run_dir>/oom_recoveries.csv``.

    @param run_dir - The run directory created by ``start_run``.
    @param resume - Keep the events earlier attempts of the same run wrote.
    @returns Nothing; events recorded before the bind are written now.
    """
    _RECORDER.bind(run_dir, resume=resume)


def record_recovery(**row: Any) -> None:
    """Record one OOM recovery event (see the module table for the columns).

    @param row - Column values; unknown keys are ignored, missing ones left empty.
    @returns Nothing; the CSV is rewritten and the event logged at WARNING.
    """
    _RECORDER.record(row)


def recovery_rows() -> list[dict[str, Any]]:
    """Return a copy of every event recorded in this process.

    @returns One dict per event, keyed by :data:`COLUMNS`.
    """
    return _RECORDER.rows()


def reset_for_tests() -> None:
    """Clear all recorded events and the bound run directory (tests only)."""
    _RECORDER.reset()
