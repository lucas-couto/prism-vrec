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
from src.evaluation.persistence import cell_key as _artifact_key
from src.evaluation.persistence import read_cell_artifact

STATES = ("pending", "running", "done", "failed")
_MANIFEST_VERSION = 1


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
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(path=path, cells=data.get("cells", {}))
        return cls(path=path, cells={})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _MANIFEST_VERSION, "updated_at": _now(), "cells": self.cells}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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


def is_cell_complete(cell: BatteryCell, results_dir: str | Path) -> bool:
    """Idempotency: True if the cell's per-user artifact (F) exists + validates."""
    key = _artifact_key(cell.dataset, cell.visual_config, cell.recommender, cell.seed)
    records_path = Path(results_dir) / "per_user" / cell.dataset / f"{key}.csv.gz"
    meta_path = records_path.with_name(records_path.name.replace(".csv.gz", ".meta.json"))
    if not (records_path.exists() and meta_path.exists()):
        return False
    try:
        _, df = read_cell_artifact(records_path)
    except Exception:  # noqa: BLE001 — a corrupt/partial artifact is not complete
        return False
    return not df.empty and {"user_idx", "rank"}.issubset(df.columns)


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
