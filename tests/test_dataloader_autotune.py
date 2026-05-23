"""Tests for the DataLoader autotune helper.

The framework picks ``num_workers`` / ``prefetch_factor`` /
``batch_size`` from a 3-tier table indexed by the container's memory
budget; the values are the contract that every step that builds a
DataLoader relies on.  These tests pin both the *tier boundaries*
(a small change to the thresholds is caught here) and the *escape
hatches* (env-var overrides, CPU clamp, cgroup parsing).

The module under test caches via ``@lru_cache``, every test clears
the cache before exercising new inputs.
"""

from __future__ import annotations

import pytest

from src.utils import dataloader as autotune_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    """Wipe ``autotune``'s memoization before AND after every test.

    Otherwise the first test's monkeypatched values would leak into
    every subsequent test because :func:`autotune` is decorated with
    ``@lru_cache(maxsize=1)``.
    """
    autotune_mod.autotune.cache_clear()
    yield
    autotune_mod.autotune.cache_clear()


@pytest.fixture
def fake_host(monkeypatch):
    """Return a callable that fakes ``cpu_count`` and the memory budget.

    Usage::

        def test_x(fake_host):
            fake_host(cpu=8, memory_gb=16)
            assert autotune().settings.num_workers == 4
    """

    def _set(*, cpu: int, memory_gb: float) -> None:
        monkeypatch.setattr(autotune_mod.os, "cpu_count", lambda: cpu)
        monkeypatch.setattr(
            autotune_mod,
            "_memory_budget_bytes",
            lambda: int(memory_gb * 1024**3),
        )

    return _set


def test_tight_tier_for_small_memory(fake_host):
    fake_host(cpu=10, memory_gb=4.0)

    settings = autotune_mod.autodetect()

    assert settings.num_workers == 2
    assert settings.prefetch_factor == 2
    assert settings.batch_size == 32


def test_balanced_tier_for_mid_memory(fake_host):
    fake_host(cpu=10, memory_gb=16.0)

    settings = autotune_mod.autodetect()

    assert settings.num_workers == 4
    assert settings.prefetch_factor == 4
    assert settings.batch_size == 128


def test_loose_tier_for_large_memory(fake_host):
    fake_host(cpu=16, memory_gb=64.0)

    settings = autotune_mod.autodetect()

    assert settings.num_workers == 12
    assert settings.prefetch_factor == 8
    assert settings.batch_size == 256


@pytest.mark.parametrize(
    ("memory_gb", "expected_tier"),
    [
        (7.99, "tight (<8 GB)"),
        (8.0, "balanced (8-32 GB)"),
        (31.99, "balanced (8-32 GB)"),
        (32.0, "loose (>=32 GB)"),
    ],
)
def test_tier_boundaries_inclusive_on_upper_edge(fake_host, memory_gb, expected_tier):
    fake_host(cpu=16, memory_gb=memory_gb)

    assert autotune_mod.autotune().tier_name == expected_tier


def test_num_workers_clamped_to_cpu_minus_one(fake_host):
    # Loose tier wants 12 workers, but only 3 cores available.
    fake_host(cpu=3, memory_gb=64.0)

    settings = autotune_mod.autodetect()

    assert settings.num_workers == 2  # min(12, 3 - 1)


def test_single_core_host_still_yields_valid_workers(fake_host):
    fake_host(cpu=1, memory_gb=64.0)

    settings = autotune_mod.autodetect()

    # cpu - 1 == 0 but we floor at 1 so persistent_workers stays valid.
    assert settings.num_workers >= 1


def test_zero_cpu_fallback(fake_host, monkeypatch):
    # os.cpu_count() returns None on some exotic environments.
    monkeypatch.setattr(autotune_mod.os, "cpu_count", lambda: None)
    monkeypatch.setattr(
        autotune_mod,
        "_memory_budget_bytes",
        lambda: int(16 * 1024**3),
    )

    settings = autotune_mod.autodetect()

    assert settings.num_workers >= 1


def test_yaml_override_replaces_num_workers(fake_host):
    fake_host(cpu=10, memory_gb=16.0)

    settings = autotune_mod.resolve_dataloader_settings(
        {"dataloader": {"num_workers": 9}},
    )

    assert settings.num_workers == 9
    # The other fields fall through to the auto values.
    assert settings.prefetch_factor == 4
    assert settings.batch_size == 128


def test_yaml_override_replaces_prefetch_and_batch_size(fake_host):
    fake_host(cpu=10, memory_gb=16.0)

    settings = autotune_mod.resolve_dataloader_settings(
        {"dataloader": {"prefetch_factor": 16, "batch_size": 512}},
    )

    assert settings.prefetch_factor == 16
    assert settings.batch_size == 512
    assert settings.num_workers == 4  # untouched


def test_yaml_overrides_absent_falls_through_to_auto(fake_host):
    fake_host(cpu=10, memory_gb=4.0)

    settings = autotune_mod.resolve_dataloader_settings(None)

    assert (settings.num_workers, settings.prefetch_factor, settings.batch_size) == (2, 2, 32)


def test_yaml_none_values_fall_through_to_auto(fake_host):
    """A pydantic-validated config has the keys with None as default,
    which must behave the same as missing keys."""
    fake_host(cpu=10, memory_gb=16.0)

    settings = autotune_mod.resolve_dataloader_settings(
        {"dataloader": {"num_workers": None, "prefetch_factor": None, "batch_size": None}},
    )

    assert (settings.num_workers, settings.prefetch_factor, settings.batch_size) == (4, 4, 128)


def test_describe_returns_expected_shape_without_overrides(fake_host):
    fake_host(cpu=10, memory_gb=16.0)

    snapshot = autotune_mod.describe()

    assert snapshot == {
        "cpu_count": 10,
        "memory_budget_gb": 16.0,
        "tier": "balanced (8-32 GB)",
        "auto": {"num_workers": 4, "prefetch_factor": 4, "batch_size": 128},
        "resolved": {"num_workers": 4, "prefetch_factor": 4, "batch_size": 128},
        "yaml_overrides": {},
    }


def test_describe_captures_active_yaml_overrides(fake_host):
    fake_host(cpu=10, memory_gb=16.0)

    snapshot = autotune_mod.describe({"dataloader": {"num_workers": 9}})

    # Auto values are reported as-is, the resolved block reflects the
    # override, and yaml_overrides surfaces the pinned key.
    assert snapshot["auto"]["num_workers"] == 4
    assert snapshot["resolved"]["num_workers"] == 9
    assert snapshot["yaml_overrides"] == {"num_workers": 9}


def test_cgroup_no_limit_sentinel_is_ignored(fake_host, monkeypatch, tmp_path):
    """A huge cgroup-v1 limit value means 'no limit', not 'plenty of RAM'.

    The kernel reports something close to ``2 ** 63`` when no limit
    has been set; we must fall through to the next budget source.
    """
    # v2 unset, v1 reports the sentinel, host reports 12 GB.
    monkeypatch.setattr(
        autotune_mod,
        "_read_int_file",
        lambda p: (1 << 63) - 4096 if "memory.limit_in_bytes" in str(p) else None,
    )
    monkeypatch.setattr(
        autotune_mod.os,
        "sysconf",
        lambda name: 1024 if name == "SC_PAGE_SIZE" else (12 * 1024**3 // 1024),
    )

    assert autotune_mod._memory_budget_bytes() == 12 * 1024**3
