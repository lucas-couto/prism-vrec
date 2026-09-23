"""Append a dataset's pre-training cost to ``results/pre_training_costs.json``.

The pipeline records per-cell costs inside each run directory, but
nothing writes the cost of the steps that precede training. This reads a
finished run's ``step_timings.json`` and ``manifest.json`` and appends
the download / preprocess / extract / fuse figures for one dataset,
keeping whatever is already in the file.

    python scripts/record_pretraining_cost.py <dataset> <run_dir>

The download figure comes from ``data/raw/host_download_timings.json``
(measured on the host, outside the container, because the server drops
the connection on long transfers). A step the run skipped — its
artifacts were already on disk — is reported as skipped rather than as
a zero, so a missing measurement is never mistaken for a cheap step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COSTS = Path("results/pre_training_costs.json")
DOWNLOAD_RECORD = Path("data/raw/host_download_timings.json")
PRE_TRAIN_STEPS = ("preprocess", "extract", "fuse")


def stage_from_timings(timings: list[dict], step: str, dataset: str) -> dict | None:
    """Aggregate every cell of *step* for *dataset* into one stage record."""
    cells = [
        t
        for t in timings
        if t.get("step", "").split(" ")[0] == step
        and (t.get("labels") or {}).get("dataset") in (dataset, None)
    ]
    if not cells:
        return None
    seconds = sum(float(t.get("duration_seconds") or 0) for t in cells)
    energy = sum(
        float(((t.get("telemetry") or {}).get("cost") or {}).get("energy_wh") or 0)
        for t in cells
    )
    skipped = [t for t in cells if t.get("skipped")]
    return {
        "cells": len(cells),
        "skipped_cells": len(skipped),
        "wall_seconds": round(seconds, 1),
        "energy_wh": round(energy, 2),
        # An explicit flag, never a prose note: a stage can carry a
        # descriptive note AND be a real measurement, and an earlier
        # version of this script dropped a measured stage from the total
        # because it had a note attached.
        "skipped": len(skipped) == len(cells),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    dataset, run_dir = argv[0], Path(argv[1])

    timings = json.loads((run_dir / "step_timings.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())

    doc = json.loads(COSTS.read_text()) if COSTS.exists() else {
        "schema": "pre_training_costs/1",
        "note": "Cost of the steps that precede training.",
        "datasets": {},
    }
    entry = doc["datasets"].setdefault(dataset, {})
    entry["run_id"] = manifest.get("run_id")
    entry["code_version"] = (manifest.get("git") or {}).get("short_sha")

    for step in PRE_TRAIN_STEPS:
        stage = stage_from_timings(timings, step, dataset)
        if stage is not None:
            entry[step] = stage

    if DOWNLOAD_RECORD.exists():
        record = json.loads(DOWNLOAD_RECORD.read_text())["datasets"].get(dataset)
        if record:
            entry["download"] = {
                "bytes": record["bytes"],
                "duration_seconds": record["duration_seconds"],
                "mean_mb_per_s": record.get("mean_mb_per_s"),
                "connection_drops": record.get("connection_drops"),
                "measured_by": "host wget (see data/raw/host_download_timings.json)",
            }

    measured = [
        entry[s]["wall_seconds"]
        for s in PRE_TRAIN_STEPS
        if isinstance(entry.get(s), dict) and not entry[s].get("skipped")
    ]
    entry["total"] = {
        "wall_seconds": round(sum(measured) + entry.get("download", {}).get("duration_seconds", 0), 1),
        "energy_wh": round(
            sum(entry[s].get("energy_wh", 0) for s in PRE_TRAIN_STEPS if isinstance(entry.get(s), dict)),
            2,
        ),
    }

    COSTS.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
