"""Per-cell cost table: one row per cell with time, RAM, GPU, VRAM, power, energy.

Every step records its cells into ``step_timings.json``; the same
entries are flattened into ``cell_costs.csv`` so the cost of each
extractor x model x dataset combination can be read without walking
nested telemetry blocks.
"""

from __future__ import annotations

import csv
import io
import json
import time

import pytest

from src.steps import fuse as fuse_mod
from src.utils import parallel, telemetry, timing
from src.utils.cell_costs import COLUMNS, cost_row, render_csv

TELEMETRY = {
    "cost": {
        "rss_mb": {"mean": 2048.0},
        "gpu_util_percent": {"mean": 46.5},
        "gpu_mem_mb": {"mean": 4096.0},
        "gpu_power_watts": {"mean": 120.0},
        "energy_wh": 5.0,
    }
}


def _entry(**labels) -> dict:
    return {
        "step": "train",
        "started_at": "2026-09-12T20:00:00Z",
        "duration_seconds": 150.0,
        "labels": labels,
        "telemetry": TELEMETRY,
    }


@pytest.fixture
def recorder(tmp_path):
    timing.reset_for_tests()
    timing.bind_run_dir(tmp_path)
    yield tmp_path
    timing.reset_for_tests()


def _rows(run_dir) -> list[dict]:
    return list(csv.DictReader(io.StringIO((run_dir / "cell_costs.csv").read_text())))


class TestCostRow:
    def test_should_convert_the_six_metrics_to_the_requested_units(self):
        row = cost_row(_entry(dataset="amazon_men"))

        assert row["time_s"] == 150.0
        assert row["ram_mean_gb"] == 2.0
        assert row["gpu_util_mean_pct"] == 46.5
        assert row["vram_mean_gb"] == 4.0
        assert row["power_mean_w"] == 120.0
        assert row["energy_wh"] == 5.0

    def test_should_join_extractors_and_keep_other_labels_as_details(self):
        row = cost_row(
            _entry(extractors=["resnet50", "vit_b16"], fusion="concat", model="vbpr", lr=0.001)
        )

        assert row["extractors"] == "resnet50+vit_b16"
        assert row["fusion"] == "concat"
        assert row["model"] == "vbpr"
        assert json.loads(row["details"]) == {"lr": 0.001}

    def test_should_map_the_singular_extractor_label_to_the_column(self):
        row = cost_row(_entry(dataset="tradesy", extractor="cvt_13"))

        assert row["extractors"] == "cvt_13"
        assert row["details"] == ""

    def test_should_leave_unmeasured_metrics_empty_instead_of_zero(self):
        entry = _entry(dataset="amazon_men")
        del entry["telemetry"]

        text = render_csv([entry])

        row = next(csv.DictReader(io.StringIO(text)))
        assert list(row) == list(COLUMNS)
        assert row["ram_mean_gb"] == ""
        assert row["energy_wh"] == ""


class TestRecorderWritesTheTable:
    def test_should_write_a_row_for_every_timed_cell(self, recorder):
        with timing.time_cell("extract", dataset="amazon_men", extractor="resnet50"):
            pass

        rows = _rows(recorder)

        assert [(r["step"], r["dataset"], r["extractors"]) for r in rows] == [
            ("extract", "amazon_men", "resnet50")
        ]

    def test_should_leave_no_row_for_a_skipped_cell(self, recorder):
        with timing.time_cell("extract", dataset="amazon_men") as cell:
            cell.skip("exists")

        assert not (recorder / "cell_costs.csv").exists()

    def test_should_record_an_external_cell_with_its_measured_duration(self, recorder):
        timing.record_cell("train", 42.5, dataset="amazon_men", model="bpr")

        (entry,) = timing.cell_timings()
        assert entry["duration_seconds"] == 42.5
        assert entry["labels"] == {"dataset": "amazon_men", "model": "bpr"}
        assert _rows(recorder)[0]["time_s"] == "42.5"

    def test_should_slice_the_external_window_from_the_parent_sampler(self, recorder):
        telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
        try:
            time.sleep(0.3)
            timing.record_cell("train", 0.25, dataset="amazon_men")
        finally:
            telemetry.stop()

        (entry,) = timing.cell_timings()
        assert entry["telemetry"]["samples"] >= 2


def _job(**overrides) -> parallel.TrainingJob:
    fields = {
        "dataset_name": "amazon_men",
        "model_name": "vbpr",
        "embedding_name": "resnet50",
        "hyperparams": {"learning_rate": 0.001},
        "n_users": 1,
        "n_items": 1,
        "embeddings_path": None,
        "processed_dir": "",
        "device": "cpu",
    }
    return parallel.TrainingJob(**{**fields, **overrides})


class TestTrainCosts:
    def test_should_record_every_attempt_with_model_extractor_and_status(self, recorder):
        parallel._record_job_cost(
            _job(), {"status": "oom", "attempt": 1, "duration": 30.0}, workers=1
        )

        (entry,) = timing.cell_timings()
        labels = entry["labels"]
        assert (labels["dataset"], labels["model"], labels["extractors"]) == (
            "amazon_men",
            "vbpr",
            ["resnet50"],
        )
        assert (labels["status"], labels["attempt"]) == ("oom", 1)
        assert labels["hyperparams"] == {"learning_rate": 0.001}

    def test_should_skip_messages_without_a_measured_duration(self, recorder):
        parallel._record_job_cost(_job(), {"status": "error"}, workers=1)

        assert timing.cell_timings() == []


class TestFuseCosts:
    def test_should_record_a_fusion_that_did_work(self, recorder, tmp_path):
        task = {
            "strategy_name": "concat",
            "output_path": str(tmp_path / "amazon_men" / "hybrid_concat.npy"),
            "emb_list_paths": [str(tmp_path / "resnet50.npy"), str(tmp_path / "vit_b16.npy")],
        }

        fuse_mod._record_fusion_cost(task, "concat: done", 12.0, n_workers=2)

        (entry,) = timing.cell_timings()
        assert entry["labels"]["dataset"] == "amazon_men"
        assert entry["labels"]["extractors"] == ["resnet50", "vit_b16"]
        assert entry["labels"]["fusion"] == "concat"
        assert entry["labels"]["concurrent_workers"] == 2

    def test_should_count_an_existing_output_as_skipped(self, recorder):
        fuse_mod._record_fusion_cost({"strategy_name": "concat"}, None, 0.0, n_workers=1)

        assert timing.cell_counts() == (0, 1)
