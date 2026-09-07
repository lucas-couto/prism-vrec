"""Battery state manifest: per-cell state, resume, idempotency, cost (Task I).

Spot instances die without warning.  The manifest is an inspectable JSON
file (``<results>/battery/manifest.json``) tracking every cell's state
(pending/running/done/failed) plus the metadata a result needs to be
traceable months later.  Idempotency: a cell whose per-user artifact (F)
exists and validates is skipped.  Cost projection reads it to estimate
the remaining wall time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.battery.cells import BatteryCell
from src.evaluation.persistence import ArtifactIntegrityError, validate_cell_artifact
from src.evaluation.persistence import cell_key as _artifact_key
from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger

logger = get_logger(__name__)

STATES = ("pending", "running", "done", "failed")
_MANIFEST_VERSION = 1

#: Metadata fields a ``done`` entry's artifact must agree with the cell on.
_CELL_FIELDS = ("dataset", "visual_config", "recommender", "seed")


class ManifestError(RuntimeError):
    """The manifest file cannot be read; it is never guessed or rebuilt silently."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class BatteryManifest:
    """A JSON-backed map ``cell_key -> {state, ...}``."""

    path: Path
    cells: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> BatteryManifest:
        path = Path(path)
        if not path.exists():
            return cls(path=path, cells={})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ManifestError(f"{path}: unreadable manifest ({exc}); refusing to guess.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("cells", {}), dict):
            raise ManifestError(f"{path}: manifest has no cells mapping.")
        return cls(path=path, cells=data.get("cells", {}))

    def save(self) -> None:
        """Publish the manifest atomically (fsync + rename); never a torn file (E07)."""
        payload = {"version": _MANIFEST_VERSION, "updated_at": _now(), "cells": self.cells}
        text = json.dumps(payload, indent=2)
        atomic_write(lambda tmp: Path(tmp).write_text(text, encoding="utf-8"), self.path)

    def sync_cells(self, cells: list[BatteryCell]) -> None:
        """Register any not-yet-seen cell as ``pending`` (idempotent)."""
        for cell in cells:
            self.cells.setdefault(
                cell.key(),
                {
                    "state": "pending",
                    "role": cell.role,
                    "dataset": cell.dataset,
                    "visual_config": cell.visual_config,
                    "recommender": cell.recommender,
                    "seed": cell.seed,
                    "error": None,
                    "duration_seconds": None,
                },
            )

    def set_state(self, key: str, state: str, **meta) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}; valid: {STATES}")
        entry = self.cells.setdefault(key, {})
        entry["state"] = state
        entry["updated_at"] = _now()
        entry.update(meta)

    def state_of(self, key: str) -> str:
        return self.cells.get(key, {}).get("state", "pending")

    def by_state(self, state: str) -> list[str]:
        return [k for k, v in self.cells.items() if v.get("state") == state]

    def summary(self) -> dict[str, int]:
        counts = {s: 0 for s in STATES}
        for entry in self.cells.values():
            counts[entry.get("state", "pending")] = counts.get(entry.get("state", "pending"), 0) + 1
        return counts


class IncompleteRunError(RuntimeError):
    """A battery / K-fold manifest still holds cells that did not finish.

    The runners return their manifest even when cells failed; without
    this check ``main.py`` exited zero on a battery with failed cells
    (audit F04).  The manifest itself is untouched, so ``--battery
    --retry-failed`` (or the next ``evaluate`` step under K-fold)
    resumes exactly the cells listed here.
    """


def require_complete(manifest: BatteryManifest, *, label: str) -> None:
    """Raise :class:`IncompleteRunError` unless every cell is ``done``.

    :param manifest: The manifest a runner returned.
    :param label: Human name of the run for the error message.
    :raises IncompleteRunError: With the per-state breakdown of unfinished cells.
    """
    summary = manifest.summary()
    unfinished = {state: n for state, n in summary.items() if state != "done" and n > 0}
    if not unfinished:
        return
    breakdown = ", ".join(f"{n} {state}" for state, n in sorted(unfinished.items()))
    raise IncompleteRunError(
        f"{label} finished with unfinished cells ({breakdown}); "
        f"{summary.get('done', 0)} done. See the manifest for the cell list."
    )


def cell_records_path(cell: BatteryCell, results_dir: str | Path) -> Path:
    """Canonical per-user records path of *cell* under *results_dir*."""
    key = _artifact_key(cell.dataset, cell.visual_config, cell.recommender, cell.seed)
    return Path(results_dir) / "per_user" / cell.dataset / f"{key}.csv.gz"


def is_cell_complete(cell: BatteryCell, results_dir: str | Path) -> bool:
    """Idempotency: True only if the cell's per-user artifact (F) is a validated generation.

    The completion pointer must exist, match the records payload byte
    for byte and declare at least one row (E06/E07).  A legacy artifact
    without a completion block, a torn pair or a missing file is not
    complete — it is recomputed, never skipped.
    """
    records_path = cell_records_path(cell, results_dir)
    if not records_path.exists():
        return False
    try:
        completion = validate_cell_artifact(records_path)
    except ArtifactIntegrityError:
        return False
    return completion is not None and completion.row_count > 0


def artifact_binding(cell: BatteryCell, results_dir: str | Path) -> dict | None:
    """What a ``done`` entry records about the artifact it stands for (E07).

    ``None`` when the cell has no validated, complete artifact.  The
    binding carries the payload digest, generation id, row count, the
    identity digest of the evaluation and the provenance fields the
    artifact's metadata declares.
    """
    records_path = cell_records_path(cell, results_dir)
    if not records_path.exists():
        return None
    try:
        completion = validate_cell_artifact(records_path)
    except ArtifactIntegrityError:
        return None
    if completion is None or completion.row_count <= 0:
        return None
    meta_path = records_path.with_name(records_path.name.replace(".csv.gz", ".meta.json"))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "records": str(records_path),
        "records_sha256": completion.records_sha256,
        "generation_id": completion.generation_id,
        "row_count": completion.row_count,
        "identity_digest": completion.identity_digest,
        "provenance": {
            **{k: metadata.get(k) for k in _CELL_FIELDS},
            "split": metadata.get("split"),
            "eval_protocol_version": metadata.get("eval_protocol_version"),
            "n_items": metadata.get("n_items"),
            "config_hash": metadata.get("config_hash"),
        },
    }


def _provenance_matches_cell(binding: dict, cell: BatteryCell) -> str | None:
    provenance = binding.get("provenance") or {}
    for field_name in _CELL_FIELDS:
        wanted = getattr(cell, field_name)
        if str(provenance.get(field_name)) != str(wanted):
            return f"artifact {field_name}={provenance.get(field_name)!r} != cell {wanted!r}"
    return None


def done_entry_valid(entry: dict, cell: BatteryCell, results_dir: str | Path) -> tuple[bool, str]:
    """Whether a ``done`` manifest entry still stands for a valid artifact (E07).

    Valid only when the entry carries an ``artifact`` binding, the
    artifact on disk validates as complete, its payload digest and
    generation equal the recorded ones, and its provenance names this
    cell.  A legacy entry without a binding, a missing/torn artifact, a
    replaced payload or another cell's artifact is *not* done; the reason
    is returned for the log and the manifest note.
    """
    recorded = entry.get("artifact")
    if not isinstance(recorded, dict):
        return False, "legacy done entry without an artifact binding"
    current = artifact_binding(cell, results_dir)
    if current is None:
        return False, "artifact missing, torn or without a validated completion"
    for key in ("records_sha256", "generation_id", "row_count"):
        if current.get(key) != recorded.get(key):
            return False, f"artifact {key} differs from the recorded binding"
    if (current.get("provenance") or {}) != (recorded.get("provenance") or {}):
        return False, "artifact provenance differs from the recorded binding"
    mismatch = _provenance_matches_cell(current, cell)
    if mismatch:
        return False, mismatch
    return True, "artifact validated"


def present_artifact_binding(cell: BatteryCell, results_dir: str | Path) -> dict | None:
    """Binding of a validated artifact that names *cell*, or ``None``."""
    binding = artifact_binding(cell, results_dir)
    if binding is None:
        return None
    mismatch = _provenance_matches_cell(binding, cell)
    if mismatch:
        logger.warning("%s: artifact present but %s; not reused.", cell.key(), mismatch)
        return None
    return binding


def project_cost(manifest: BatteryManifest) -> dict:
    """Estimate remaining wall time from completed-cell durations.

    Uses the mean duration per cell role (search vs replay differ a lot),
    times the pending cells of each role.  Silent about roles with no
    completed sample yet — reports them as unknown rather than guessing.
    """
    done_durations: dict[str, list[float]] = {}
    pending_by_role: dict[str, int] = {}
    for entry in manifest.cells.values():
        role = entry.get("role", "search")
        if entry.get("state") == "done" and entry.get("duration_seconds") is not None:
            done_durations.setdefault(role, []).append(float(entry["duration_seconds"]))
        elif entry.get("state") in ("pending", "failed", "running"):
            pending_by_role[role] = pending_by_role.get(role, 0) + 1

    est_seconds = 0.0
    unknown_roles = []
    for role, n_pending in pending_by_role.items():
        samples = done_durations.get(role)
        if samples:
            est_seconds += (sum(samples) / len(samples)) * n_pending
        else:
            unknown_roles.append(role)

    return {
        "summary": manifest.summary(),
        "estimated_remaining_hours": round(est_seconds / 3600, 2),
        "roles_without_estimate": unknown_roles,
    }
