"""Pools must size off the cgroup's CPU quota, not the machine's cores.

A container limited with ``cpus: 10.4`` still sees 16 cores through
``os.cpu_count()``; a pool sized from that oversubscribes its own quota
and spends the difference on context switching, which on a workstation
is felt as a frozen desktop.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.dataloader import resolve_dataloader_settings
from src.utils.memory import available_cpus

#: Modules that size a process pool and must therefore honour the quota.
_POOL_SIZING_MODULES = (
    "src/utils/parallel.py",
    "src/utils/dataloader.py",
    "src/steps/fuse.py",
)


def test_should_report_at_least_one_usable_core() -> None:
    assert available_cpus() >= 1


def test_should_never_exceed_the_machines_core_count() -> None:
    import os

    assert available_cpus() <= (os.cpu_count() or 1)


def test_should_clamp_a_pinned_num_workers_to_the_quota() -> None:
    absurd = {"resources": {"workers": {"dataloader": 512}}}

    settings = resolve_dataloader_settings(absurd)

    assert settings.num_workers <= max(1, available_cpus() - 1)


def test_should_honour_a_pinned_num_workers_that_fits() -> None:
    settings = resolve_dataloader_settings({"resources": {"workers": {"dataloader": 1}}})

    assert settings.num_workers == 1


def test_should_not_size_pools_from_os_cpu_count() -> None:
    offenders = [
        path
        for path in _POOL_SIZING_MODULES
        if "os.cpu_count()" in Path(path).read_text(encoding="utf-8")
    ]

    assert offenders == [], f"size pools with available_cpus(): {offenders}"
