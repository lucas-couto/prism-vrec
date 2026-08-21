"""Tests for per-step throughput / cost telemetry."""

from __future__ import annotations

import time

import pytest

from src.utils import telemetry


@pytest.fixture(autouse=True)
def _clean_sampler():
    """Guarantee no sampler leaks from one test into the next."""
    telemetry.reset_for_tests()
    yield
    telemetry.reset_for_tests()


def _busy_wait(seconds: float) -> None:
    """Sleep in small slices so the sampler thread definitely runs."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        time.sleep(0.005)


def test_should_return_none_summary_when_telemetry_is_disabled():
    telemetry.start({"telemetry": {"enabled": False}})

    marker = telemetry.mark()

    assert marker is None
    assert telemetry.summarise_since(marker) is None
    assert telemetry.is_active() is False


def test_should_report_network_throughput_from_recorded_bytes():
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    marker = telemetry.mark()

    for _ in range(10):
        telemetry.add_bytes(1024 * 1024)
        _busy_wait(0.05)
    summary = telemetry.summarise_since(marker)

    throughput = summary["throughput"]
    assert throughput["total_bytes"] == 10 * 1024 * 1024
    assert throughput["network_mb_per_s"]["mean"] > 0


def test_should_report_min_max_and_mean_for_a_sampled_window():
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    marker = telemetry.mark()

    # Uneven work: an idle stretch then a burst, so min != max.
    _busy_wait(0.2)
    for _ in range(5):
        telemetry.add_items(100)
        _busy_wait(0.05)
    summary = telemetry.summarise_since(marker)

    rate = summary["throughput"]["items_per_s"]
    assert rate["min"] <= rate["mean"] <= rate["max"]
    assert rate["min"] < rate["max"]


def test_should_omit_counters_that_never_moved():
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    marker = telemetry.mark()

    telemetry.add_items(5)
    _busy_wait(0.2)
    summary = telemetry.summarise_since(marker)

    # Nothing was downloaded and no FLOPs were recorded, so those series
    # must be absent rather than present-and-zero.
    assert "network_mb_per_s" not in summary["throughput"]
    assert "flops_per_s" not in summary["throughput"]


def test_should_always_report_cpu_cost_even_without_a_gpu():
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    marker = telemetry.mark()

    _busy_wait(0.25)
    summary = telemetry.summarise_since(marker)

    assert "cpu_util_percent" in summary["cost"]


def test_should_still_account_for_a_step_shorter_than_one_sampling_interval():
    # A 30 s interval means the background thread never ticks during this
    # step; the forced edge samples must carry the counter delta anyway.
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 30}})
    marker = telemetry.mark()

    telemetry.add_items(42)
    summary = telemetry.summarise_since(marker)

    throughput = summary["throughput"]
    assert throughput["total_items"] == 42
    # Two samples cannot describe a distribution, so only the mean is given.
    assert "min" not in throughput["items_per_s"]


def test_should_not_start_a_second_sampler_while_one_is_running():
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    first = telemetry.probes()

    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})

    assert telemetry.is_active() is True
    assert telemetry.probes() == first


def test_should_ignore_counter_updates_when_no_sampler_is_running():
    # No start() call: the counters must be inert, not raise.
    telemetry.add_bytes(1024)
    telemetry.add_flops(1e9)
    telemetry.add_items(1)

    assert telemetry.is_active() is False
    assert telemetry.summarise_since(None) is None


def test_should_write_raw_series_when_save_samples_is_enabled(tmp_path):
    telemetry.start(
        {"telemetry": {"enabled": True, "sample_interval_seconds": 0.05, "save_samples": True}},
        run_dir=tmp_path,
    )
    telemetry.add_items(3)
    _busy_wait(0.2)

    telemetry.stop()

    series = tmp_path / "telemetry_samples.jsonl"
    assert series.exists()
    assert len(series.read_text().strip().splitlines()) >= 2


def test_should_integrate_power_into_energy_with_the_trapezoid_rule():
    # 100 W held for 2 s = 200 J; the trapezoid of a constant series is exact.
    series = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0)]

    joules = telemetry._integrate(series)

    assert joules == pytest.approx(200.0)


def test_should_report_mean_only_when_the_window_has_too_few_samples():
    stats = telemetry._stats([4.0, 6.0])

    assert stats["mean"] == pytest.approx(5.0)
    assert "min" not in stats


def test_should_still_report_probe_backends_after_the_sampler_stops():
    # The manifest is written after stop(); reporting "none" there would
    # misattribute measurements that were in fact taken.
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    live = telemetry.probes()

    telemetry.stop()

    assert telemetry.probes() == live
    assert live["cpu"] != "none"
