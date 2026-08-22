"""Tests for the per-dataset timing cells emitted by ``steps.download``.

The download step used to be one opaque window covering every dataset,
so "which dataset cost the download hour" was unanswerable and an
interrupted download left no trace at all.  Each dataset is now its own
cell in ``step_timings.json``, carrying its weight on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.base import DatasetProvider, register_dataset_provider
from src.steps import download
from src.utils import timing


class _FakeProvider(DatasetProvider):
    """Provider that writes a fixed payload into its raw dir."""

    def __init__(self, name: str, raw_dir: Path, payload_bytes: int) -> None:
        super().__init__(name=name, raw_dir=raw_dir)
        self._payload_bytes = payload_bytes

    def download(self) -> None:
        (self.raw_dir / "archive.bin").write_bytes(b"x" * self._payload_bytes)

    def save_processed(self, processed_dir) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def extract_images(self, image_dir) -> None:  # pragma: no cover - unused
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _isolated_recorder():
    timing.reset_for_tests()
    yield
    timing.reset_for_tests()


@pytest.fixture
def _fake_datasets(tmp_path, monkeypatch):
    """Register two providers of different weights and configure them."""

    def _register(name: str, payload_bytes: int) -> None:
        register_dataset_provider(
            name, lambda n=name, b=payload_bytes: _FakeProvider(n, tmp_path, b)
        )

    _register("light_ds", 1_000_000)
    _register("heavy_ds", 3_000_000)
    monkeypatch.setattr(
        download, "load_config", lambda: {"datasets": ["light_ds", "heavy_ds"]}
    )


def test_each_dataset_gets_its_own_cell(_fake_datasets):
    download.run()

    cells = timing.cell_timings()
    assert [c["step"] for c in cells] == ["download", "download"]
    assert [c["labels"]["dataset"] for c in cells] == ["light_ds", "heavy_ds"]


def test_cell_carries_the_dataset_weight_on_disk(_fake_datasets):
    download.run()

    by_name = {c["labels"]["dataset"]: c["labels"] for c in timing.cell_timings()}
    assert by_name["light_ds"]["size_mb"] == 1.0
    assert by_name["heavy_ds"]["size_mb"] == 3.0


def test_downloaded_mb_is_zero_when_the_data_was_already_there(_fake_datasets):
    download.run()
    timing.reset_for_tests()

    download.run()

    for labels in (c["labels"] for c in timing.cell_timings()):
        assert labels["downloaded_mb"] == 0.0
        assert labels["size_mb"] > 0.0


def test_first_run_reports_the_bytes_it_fetched(_fake_datasets):
    download.run()

    by_name = {c["labels"]["dataset"]: c["labels"] for c in timing.cell_timings()}
    assert by_name["light_ds"]["downloaded_mb"] == 1.0
    assert by_name["heavy_ds"]["downloaded_mb"] == 3.0


def test_an_already_present_dataset_is_still_timed(_fake_datasets):
    """Re-validating a large archive is real wall-time; never skipped."""
    download.run()
    timing.reset_for_tests()

    download.run()

    recorded, skipped = timing.cell_counts()
    assert (recorded, skipped) == (2, 0)


def test_cells_are_flushed_to_the_sidecar(_fake_datasets, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    timing.bind_run_dir(run_dir)

    download.run()

    import json

    payload = json.loads((run_dir / "step_timings.json").read_text())
    assert [c["labels"]["dataset"] for c in payload] == ["light_ds", "heavy_ds"]
