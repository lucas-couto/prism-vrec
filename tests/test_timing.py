"""Tests for the per-step / per-cell timing accumulator.

The recorder is a module-level singleton, so every test starts from
a clean slate via :func:`reset_for_tests` to keep the order of test
execution irrelevant.
"""

from __future__ import annotations

import json
import re
import time

import pytest

from src.utils import timing


@pytest.fixture(autouse=True)
def _isolated_recorder():
    """Reset the singleton before AND after every test."""
    timing.reset_for_tests()
    yield
    timing.reset_for_tests()


def test_record_step_appends_entry():
    timing.record_step("extract", "2026-05-14T10:00:00Z", 123.456)

    assert timing.step_timings() == [
        {
            "name": "extract",
            "started_at": "2026-05-14T10:00:00Z",
            "duration_seconds": 123.456,
        }
    ]


def test_record_step_rounds_duration_to_milliseconds():
    timing.record_step("preprocess", "2026-05-14T10:00:00Z", 1.23456789)

    [entry] = timing.step_timings()
    assert entry["duration_seconds"] == 1.235


def test_step_timings_returns_a_copy_not_a_reference():
    """Mutating the returned list must not corrupt the recorder."""
    timing.record_step("download", "2026-05-14T10:00:00Z", 1.0)

    snapshot = timing.step_timings()
    snapshot.clear()

    assert len(timing.step_timings()) == 1


def test_time_cell_captures_duration_and_labels():
    with timing.time_cell("extract", dataset="amazon_fashion", extractor="resnet50", dim=128):
        time.sleep(0.05)

    [entry] = timing.cell_timings()

    assert entry["step"] == "extract"
    assert entry["labels"] == {
        "dataset": "amazon_fashion",
        "extractor": "resnet50",
        "dim": 128,
    }
    assert entry["duration_seconds"] >= 0.04
    # 1.0 s upper bound catches a degenerate clock; perf_counter
    # measures sub-second.
    assert entry["duration_seconds"] < 1.0


def test_time_cell_records_started_at_in_iso_format():
    with timing.time_cell("finetune", dataset="X", extractor="Y"):
        pass

    [entry] = timing.cell_timings()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", entry["started_at"])


def test_time_cell_records_even_on_exception():
    """Try/finally guarantees the duration is captured when the body raises."""
    with pytest.raises(ValueError, match="boom"), timing.time_cell("extract", x=1):
        raise ValueError("boom")

    [entry] = timing.cell_timings()
    assert entry["labels"] == {"x": 1}


def test_time_cell_multiple_calls_accumulate_in_order():
    with timing.time_cell("extract", extractor="a"):
        pass
    with timing.time_cell("extract", extractor="b"):
        pass

    names = [c["labels"]["extractor"] for c in timing.cell_timings()]
    assert names == ["a", "b"]


def test_time_cell_writes_sidecar_json_after_bind(tmp_path):
    timing.bind_run_dir(tmp_path)

    with timing.time_cell("extract", dataset="ds", extractor="vit"):
        pass

    sidecar = tmp_path / "step_timings.json"
    assert sidecar.exists()

    payload = json.loads(sidecar.read_text())
    assert isinstance(payload, list)
    assert payload[0]["step"] == "extract"
    assert payload[0]["labels"] == {"dataset": "ds", "extractor": "vit"}


def test_time_cell_skips_disk_when_not_bound(tmp_path):
    """Without bind_run_dir, the cell is still recorded in memory but
    no sidecar is written, useful for unit tests and smoke runs that
    do not produce a manifest.
    """
    with timing.time_cell("extract", x=1):
        pass

    assert not (tmp_path / "step_timings.json").exists()
    assert len(timing.cell_timings()) == 1


def test_bind_overwrites_previous_run_dir(tmp_path):
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    first.mkdir()
    second.mkdir()

    timing.bind_run_dir(first)
    with timing.time_cell("extract", x=1):
        pass
    timing.bind_run_dir(second)
    with timing.time_cell("extract", x=2):
        pass

    # The recorder writes the full accumulated history on every cell,
    # so the LATER bind catches the full list.
    assert (first / "step_timings.json").exists()
    payload = json.loads((second / "step_timings.json").read_text())
    assert [c["labels"]["x"] for c in payload] == [1, 2]


def test_reset_clears_steps_cells_and_bind(tmp_path):
    timing.bind_run_dir(tmp_path)
    timing.record_step("extract", "2026-05-14T10:00:00Z", 1.0)
    with timing.time_cell("extract", x=1):
        pass

    timing.reset_for_tests()

    assert timing.step_timings() == []
    assert timing.cell_timings() == []
    # After reset the recorder is unbound, so a subsequent time_cell
    # should not try to write to the old tmp_path.
    with timing.time_cell("extract", x=2):
        pass
    assert (tmp_path / "step_timings.json").read_text()
