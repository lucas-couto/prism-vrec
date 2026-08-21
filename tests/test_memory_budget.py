"""Tests for the host-memory budget and process-pool sizing helpers.

``plan_pool_workers`` is the guard that stopped the fusion step from
asking for one worker per core regardless of how much RAM each worker
holds — the sizing bug that let a run exhaust host memory and trigger a
global OOM.  These tests pin the clamping rules and the budget
resolution order (cgroup v2 -> cgroup v1 -> host RAM -> fallback).
"""

from __future__ import annotations

from src.utils import memory as memory_mod


class TestMemoryBudget:
    def test_cgroup_v2_limit_wins_over_host_memory(self, monkeypatch):
        monkeypatch.setattr(
            memory_mod,
            "_read_int_file",
            lambda p: 8 * 1024**3 if "memory.max" in str(p) else None,
        )

        assert memory_mod.memory_budget_bytes() == 8 * 1024**3

    def test_cgroup_no_limit_sentinel_is_ignored(self, monkeypatch):
        """A huge cgroup-v1 limit value means 'no limit', not 'plenty of RAM'.

        The kernel reports something close to ``2 ** 63`` when no limit
        has been set; we must fall through to the next budget source.
        """
        monkeypatch.setattr(
            memory_mod,
            "_read_int_file",
            lambda p: (1 << 63) - 4096 if "memory.limit_in_bytes" in str(p) else None,
        )
        monkeypatch.setattr(
            memory_mod.os,
            "sysconf",
            lambda name: 1024 if name == "SC_PAGE_SIZE" else (12 * 1024**3 // 1024),
        )

        assert memory_mod.memory_budget_bytes() == 12 * 1024**3

    def test_falls_back_when_nothing_is_readable(self, monkeypatch):
        monkeypatch.setattr(memory_mod, "_read_int_file", lambda p: None)

        def _boom(name):
            raise OSError("no sysconf here")

        monkeypatch.setattr(memory_mod.os, "sysconf", _boom)

        assert memory_mod.memory_budget_bytes() == 4 * 1024**3


class TestPlanPoolWorkers:
    def _fake_budget(self, monkeypatch, gb: float) -> None:
        monkeypatch.setattr(
            memory_mod,
            "memory_budget_bytes",
            lambda: int(gb * 1024**3),
        )

    def test_memory_lowers_the_cpu_cap(self, monkeypatch):
        # 24 GB budget - 4 GB reserve = 20 GB for 8 GB workers -> 2.
        self._fake_budget(monkeypatch, 24.0)

        n = memory_mod.plan_pool_workers(
            per_worker_bytes=8 * 1024**3,
            hard_cap=12,
        )

        assert n == 2

    def test_hard_cap_wins_when_memory_is_plentiful(self, monkeypatch):
        self._fake_budget(monkeypatch, 512.0)

        n = memory_mod.plan_pool_workers(
            per_worker_bytes=1 * 1024**3,
            hard_cap=6,
        )

        assert n == 6

    def test_always_returns_at_least_one_worker(self, monkeypatch):
        """A single oversized task must still run, not deadlock at zero."""
        self._fake_budget(monkeypatch, 6.0)

        n = memory_mod.plan_pool_workers(
            per_worker_bytes=64 * 1024**3,
            hard_cap=12,
        )

        assert n == 1

    def test_unknown_footprint_leaves_the_cap_untouched(self, monkeypatch):
        self._fake_budget(monkeypatch, 4.0)

        n = memory_mod.plan_pool_workers(per_worker_bytes=0, hard_cap=9)

        assert n == 9

    def test_reserve_is_withheld_from_the_pool(self, monkeypatch):
        # Without the reserve 16 GB / 4 GB would allow 4 workers.
        self._fake_budget(monkeypatch, 16.0)

        n = memory_mod.plan_pool_workers(
            per_worker_bytes=4 * 1024**3,
            hard_cap=8,
            reserve_bytes=8 * 1024**3,
        )

        assert n == 2
