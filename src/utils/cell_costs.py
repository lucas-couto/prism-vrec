"""Flat per-cell cost table: ``results/runs/<run_id>/cell_costs.csv``.

``step_timings.json`` keeps every cell's full telemetry block, nested
and verbose.  The dissertation needs one row per cell with a fixed set
of columns, groupable by step, dataset, extractor, fusion and model:

=====================  ==================================================
column                 source in the cell's telemetry block
=====================  ==================================================
``time_s``             ``duration_seconds``
``ram_mean_gb``        ``cost.rss_mb.mean`` / 1024 (process + children)
``gpu_util_mean_pct``  ``cost.gpu_util_percent.mean`` (whole device)
``vram_mean_gb``       ``cost.gpu_mem_mb.mean`` / 1024 (whole device)
``power_mean_w``       ``cost.gpu_power_watts.mean`` (whole device)
``energy_wh``          ``cost.energy_wh`` (power integrated over the cell)
=====================  ==================================================

The GPU gauges are NVML readings of the whole card, so they include
whatever else the device was doing (a desktop session, a concurrent
worker); ``concurrent_workers`` in ``details`` flags cells whose window
was shared with other training jobs.  A metric the sampler could not
read is left empty rather than written as zero.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import Any

LABEL_COLUMNS = ("step", "dataset", "extractors", "fusion", "model", "embedding")
METRIC_COLUMNS = (
    "time_s",
    "ram_mean_gb",
    "gpu_util_mean_pct",
    "vram_mean_gb",
    "power_mean_w",
    "energy_wh",
)
COLUMNS = (*LABEL_COLUMNS, "details", "started_at", *METRIC_COLUMNS)

_MB_PER_GB = 1024.0


def _mean(cost: dict[str, Any], key: str, scale: float = 1.0) -> float | None:
    stats = cost.get(key)
    if not isinstance(stats, dict) or stats.get("mean") is None:
        return None
    return round(float(stats["mean"]) / scale, 4)


def _label_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return "+".join(str(v) for v in value)
    return str(value)


def cost_row(entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``step_timings.json`` entry into a :data:`COLUMNS` row.

    :param entry: A cell entry as written by :mod:`src.utils.timing`.
    :returns: A dict keyed by :data:`COLUMNS`; labels outside the fixed
        columns are kept as compact JSON under ``details``.
    """
    labels = dict(entry.get("labels") or {})
    cost = (entry.get("telemetry") or {}).get("cost") or {}
    # Steps that time a single backbone (download, extract, finetune)
    # label it ``extractor``; the column is the plural for every step.
    if "extractors" not in labels and "extractor" in labels:
        labels["extractors"] = [labels.pop("extractor")]
    row: dict[str, Any] = {"step": entry.get("step", "")}
    for column in LABEL_COLUMNS[1:]:
        row[column] = _label_text(labels.pop(column, None))
    row["details"] = json.dumps(labels, sort_keys=True, default=str) if labels else ""
    row["started_at"] = entry.get("started_at", "")
    row["time_s"] = entry.get("duration_seconds")
    row["ram_mean_gb"] = _mean(cost, "rss_mb", _MB_PER_GB)
    row["gpu_util_mean_pct"] = _mean(cost, "gpu_util_percent")
    row["vram_mean_gb"] = _mean(cost, "gpu_mem_mb", _MB_PER_GB)
    row["power_mean_w"] = _mean(cost, "gpu_power_watts")
    row["energy_wh"] = cost.get("energy_wh")
    return row


def render_csv(entries: Sequence[dict[str, Any]]) -> str:
    """Render cell entries as the ``cell_costs.csv`` text.

    :param entries: Cell entries in recording order.
    :returns: CSV text with a header row, one row per entry.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        writer.writerow({k: ("" if v is None else v) for k, v in cost_row(entry).items()})
    return buffer.getvalue()
