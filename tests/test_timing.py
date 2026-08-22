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


def test_record_step_writes_steps_sidecar_after_bind(tmp_path):
    """Steps with no cells at all (download) only exist in this file
    until ``finish_run`` copies the list into the manifest.
    """
    timing.bind_run_dir(tmp_path)

    timing.record_step("download", "2026-05-14T10:00:00Z", 12.5)

    sidecar = tmp_path / "steps.json"
    assert sidecar.exists()

    payload = json.loads(sidecar.read_text())
    assert payload == [
        {
            "name": "download",
            "started_at": "2026-05-14T10:00:00Z",
            "duration_seconds": 12.5,
        }
    ]


def test_steps_sidecar_accumulates_every_step_in_order(tmp_path):
    """An interrupted run must still document the steps it did run."""
    timing.bind_run_dir(tmp_path)

    timing.record_step("download", "2026-05-14T10:00:00Z", 1.0)
    timing.record_step("preprocess", "2026-05-14T10:01:00Z", 2.0)

    payload = json.loads((tmp_path / "steps.json").read_text())
    assert [entry["name"] for entry in payload] == ["download", "preprocess"]


def test_record_step_skips_disk_when_not_bound(tmp_path):
    timing.record_step("download", "2026-05-14T10:00:00Z", 1.0)

    assert not (tmp_path / "steps.json").exists()
    assert len(timing.step_timings()) == 1


def test_bind_flushes_steps_recorded_before_the_bind(tmp_path):
    timing.record_step("download", "2026-05-14T10:00:00Z", 1.0)

    timing.bind_run_dir(tmp_path)

    payload = json.loads((tmp_path / "steps.json").read_text())
    assert [entry["name"] for entry in payload] == ["download"]


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


# ---------------------------------------------------------------------
# Skipped cells: work that was already done on an earlier run must not
# be timed or costed, otherwise re-running a finished pipeline reports a
# fraction of a second and zero energy for an hour-long extraction.
# ---------------------------------------------------------------------


def test_skipped_cell_leaves_no_entry():
    with timing.time_cell("extract", dataset="tradesy", extractor="resnet50") as cell:
        cell.skip("embeddings already on disk")

    assert timing.cell_timings() == []


def test_skipped_cell_is_still_counted():
    with timing.time_cell("extract", dataset="tradesy", extractor="resnet50") as cell:
        cell.skip()

    assert timing.cell_counts() == (0, 1)


def test_unskipped_cell_is_recorded_as_before():
    with timing.time_cell("extract", dataset="tradesy", extractor="resnet50") as cell:
        assert cell.skipped is False

    assert len(timing.cell_timings()) == 1
    assert timing.cell_counts() == (1, 0)


def test_cell_label_merges_into_the_entry_labels():
    """A weight known only after the work runs still lands on the entry."""
    with timing.time_cell("download", dataset="tradesy") as cell:
        cell.label(size_mb=812.4, downloaded_mb=0.0)

    [entry] = timing.cell_timings()
    assert entry["labels"] == {
        "dataset": "tradesy",
        "size_mb": 812.4,
        "downloaded_mb": 0.0,
    }


def test_cell_label_overrides_a_placeholder_of_the_same_name():
    with timing.time_cell("download", dataset="tradesy", size_mb=None) as cell:
        cell.label(size_mb=812.4)

    [entry] = timing.cell_timings()
    assert entry["labels"]["size_mb"] == 812.4


def test_cell_label_on_a_skipped_cell_records_nothing():
    with timing.time_cell("download", dataset="tradesy") as cell:
        cell.label(size_mb=812.4)
        cell.skip("already on disk")

    assert timing.cell_timings() == []


def test_skipped_cell_is_not_written_to_the_sidecar(tmp_path):
    timing.bind_run_dir(tmp_path)

    with timing.time_cell("extract", dataset="tradesy", extractor="resnet50") as cell:
        cell.skip()

    assert not (tmp_path / "step_timings.json").exists()


def test_note_skipped_cell_counts_without_a_timer():
    timing.note_skipped_cell()
    timing.note_skipped_cell()

    assert timing.cell_counts() == (0, 2)


def test_record_step_marks_skipped_and_drops_telemetry():
    timing.record_step(
        "extract",
        "2026-05-14T10:00:00Z",
        0.4,
        {"cost": {"energy_wh": 0.0}},
        skipped=True,
    )

    [entry] = timing.step_timings()
    assert entry["skipped"] is True
    assert "telemetry" not in entry


def test_record_step_keeps_telemetry_when_not_skipped():
    timing.record_step(
        "extract",
        "2026-05-14T10:00:00Z",
        900.0,
        {"cost": {"energy_wh": 42.0}},
    )

    [entry] = timing.step_timings()
    assert entry["telemetry"] == {"cost": {"energy_wh": 42.0}}
    assert "skipped" not in entry


def test_reset_clears_the_skipped_counter():
    timing.note_skipped_cell()

    timing.reset_for_tests()

    assert timing.cell_counts() == (0, 0)
