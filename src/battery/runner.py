"""Battery runner: idempotent, resumable orchestration (Task I).

Drives the enumerated cells through a state manifest.  A cell whose
per-user artifact (F) already validates is skipped (idempotency); the
rest run via a pluggable ``execute`` callback, so the loop is testable
and the smoke test can inject a light execution.  Every result records
git sha + dirty flag and its duration, so a stray number is traceable
months later.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from src.battery.cells import BatteryCell, enumerate_cells
from src.battery.manifest import BatteryManifest, is_cell_complete, project_cost
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: A cell executor: run the cell, return extra metadata (durations, etc.).
CellExecutor = Callable[[BatteryCell, dict], dict]


def _git_meta() -> dict:
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001 — git may be absent (Docker/CI)
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else None}


def manifest_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "battery" / "manifest.json"


def run_battery(
    config: dict,
    results_dir: str | Path,
    execute: CellExecutor,
    *,
    retry_failed: bool = False,
) -> BatteryManifest:
    """Enumerate + run the battery, updating the state manifest.

    Idempotent: a cell that is ``done`` (or whose F artifact validates) is
    skipped.  ``retry_failed`` re-runs ``failed`` cells.  Safe to re-invoke
    after an interruption — completed cells are not repeated.
    """
    cells = enumerate_cells(config)
    manifest = BatteryManifest.load(manifest_path(results_dir))
    manifest.sync_cells(cells)
    manifest.save()
    git = _git_meta()

    for cell in cells:
        key = cell.key()
        state = manifest.state_of(key)
        if state == "done" or is_cell_complete(cell, results_dir):
            if state != "done":
                manifest.set_state(key, "done", note="artifact already present")
            continue
        if state == "failed" and not retry_failed:
            logger.info("Skipping failed cell (pass retry_failed=True to retry): %s", key)
            continue

        manifest.set_state(key, "running", **git)
        manifest.save()
        started = time.perf_counter()
        try:
            extra = execute(cell, config) or {}
            manifest.set_state(
                key,
                "done",
                duration_seconds=round(time.perf_counter() - started, 3),
                error=None,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 — isolate per-cell failures
            logger.error("Cell failed: %s (%s)", key, exc)
            manifest.set_state(
                key,
                "failed",
                duration_seconds=round(time.perf_counter() - started, 3),
                error=str(exc),
            )
        manifest.save()

    logger.info("Battery run finished: %s", manifest.summary())
    return manifest


def battery_status(results_dir: str | Path) -> dict:
    """Read the manifest and return counts + remaining-cost projection."""
    manifest = BatteryManifest.load(manifest_path(results_dir))
    projection = project_cost(manifest)
    logger.info(
        "Battery status: %s | est. remaining ~%.2f h%s",
        projection["summary"],
        projection["estimated_remaining_hours"],
        f" (no estimate yet for roles: {projection['roles_without_estimate']})"
        if projection["roles_without_estimate"]
        else "",
    )
    return projection
