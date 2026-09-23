"""The plan and timeline tables a run prints for whoever is watching it.

Three blocks, all rendered once and emitted line by line (see
:mod:`src.utils.tables` for why nothing redraws):

* :func:`render_plan` -- printed at startup, before any step runs: the
  resolved step order and how much work each step covers.  It answers
  "what did this YAML actually decide to do" without waiting hours to
  find out.
* :func:`render_timeline` -- printed as each step finishes: what is
  done, with elapsed time, and what is still owed.
* :func:`render_cells` -- the ``(recommender, dataset)`` breakdown of a
  training queue, with each cell's own projection.

The scale column is deliberately cheap.  The exact training job count
needs the full enumeration (nearly two minutes on the frozen grid), so
the plan reports the dimensions it can read straight off the merged
config and lets the train step report its own admitted total.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.utils.tables import format_duration, render_table

#: Steps whose scale is the dataset count alone.
_PER_DATASET = ("download", "preprocess")


def _count(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    return len(value) if isinstance(value, (list, tuple)) else 0


def step_scale(step: str, config: dict[str, Any]) -> str:
    """A short, cheap description of how much work *step* covers."""
    datasets = _count(config, "datasets")
    extractors = _count(config, "extractors_enabled")
    recommenders = _count(config, "recommenders_enabled")
    fusions = _count(config, "fusion_strategies_enabled")

    if step in _PER_DATASET:
        return f"{datasets} datasets"
    if step == "extract":
        return f"{datasets} datasets x {extractors} extractors"
    if step == "fuse":
        return f"{datasets} datasets x {fusions} strategies"
    if step == "train":
        return f"{datasets} datasets x {recommenders} recommenders (grid)"
    if step == "folds":
        folds = config.get("folds") or {}
        if not folds.get("enabled"):
            return "disabled"
        return f"{folds.get('k', '?')} folds"
    return "-"


def render_plan(
    steps: Sequence[str],
    config: dict[str, Any],
    *,
    condition: str,
    run_both: bool,
    condition_steps: Sequence[str] = (),
) -> list[str]:
    """The resolved plan as a table, for the top of the run log."""
    rows = []
    for index, step in enumerate(steps, start=1):
        note = ""
        if run_both and step in condition_steps:
            note = "runs twice: frozen + finetuned"
        rows.append([str(index), step, step_scale(step, config), note])
    title = f"Pipeline plan: {len(steps)} steps, condition={'both' if run_both else condition}"
    return render_table(
        ["#", "Step", "Scale", "Note"], rows, align=[">", "<", "<", "<"], title=title
    )


def render_timeline(
    steps: Sequence[str],
    timings: Sequence[dict[str, Any]],
    *,
    current: str | None = None,
) -> list[str]:
    """Which steps are done (with elapsed), running, or still pending.

    *timings* is :func:`src.utils.timing.step_timings`; its entries are
    keyed ``name`` / ``duration_seconds`` and the name carries the
    condition suffix (``"train (frozen)"``), so a step matches a record
    by the part before the suffix.  A step recorded as skipped had every
    cell already on disk and is reported as such rather than credited
    with work it did not do.
    """
    done: dict[str, dict[str, Any]] = {}
    for record in timings:
        label = str(record.get("name", ""))
        done.setdefault(label.split(" (")[0], record)

    rows = []
    total = 0.0
    for index, step in enumerate(steps, start=1):
        record = done.get(step)
        if record is not None:
            duration = float(record.get("duration_seconds") or 0.0)
            total += duration
            status = "skipped" if record.get("skipped") else "done"
            rows.append([str(index), step, status, format_duration(duration)])
        elif step == current:
            rows.append([str(index), step, "running", "-"])
        else:
            rows.append([str(index), step, "pending", "-"])
    title = f"Steps: {len(done)}/{len(steps)} complete, {format_duration(total)} of step time"
    return render_table(
        ["#", "Step", "Status", "Elapsed"], rows, align=[">", "<", "<", ">"], title=title
    )


def render_cells(progress, *, workers: int = 1) -> list[str]:
    """The ``(recommender, dataset)`` breakdown of a training queue."""
    rows = []
    for key, cell in sorted(progress.cells.items()):
        model, dataset = key
        mean, borrowed = progress.cell_mean(key)
        eta = None if mean is None else mean * cell.remaining / max(1, workers)
        # A borrowed mean is marked, never hidden: the run total charges
        # this cell that number, so showing a dash here would make the
        # rows and the total disagree.
        prefix = "~" if borrowed else ""
        rows.append(
            [
                model,
                dataset,
                f"{cell.finished}/{cell.total}",
                str(cell.failed),
                "-" if mean is None else f"{prefix}{mean:.0f}s",
                "-" if eta is None else f"{prefix}{format_duration(eta)}",
            ]
        )
    return render_table(
        ["Recommender", "Dataset", "Done", "Failed", "Mean", "Left"],
        rows,
        align=["<", "<", ">", ">", ">", ">"],
        title=progress.line(workers=workers),
    )
