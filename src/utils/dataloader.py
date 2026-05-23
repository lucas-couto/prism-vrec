"""Auto-detected DataLoader settings for safe-by-default execution.

PyTorch's ``DataLoader`` exposes three knobs (``num_workers``,
``prefetch_factor``, ``batch_size``) that interact non-trivially with
the host's CPU count and the cgroup memory budget.  Picking values
that fit *every* deployment the framework runs on, laptops, CI,
RunPod pods, lab servers, is impossible from a single hardcoded
default: too low wastes throughput on a 128 GB lab box; too high
gets the worker pool OOM-killed on a 16 GB laptop container.

This module replaces the hardcoded defaults with a small heuristic
that inspects the runtime environment once at startup and picks a
tier that fits.  Researchers never have to set ``FT_NUM_WORKERS``,
``FT_PREFETCH`` or ``EXTRACT_BATCH_SIZE`` by hand; the env vars are
still honoured when set, but only as a power-user override.

Tiers (memory budget refers to the cgroup limit when running in a
container, the total host RAM otherwise):

==============  ===========  ===========  ============
memory budget   num_workers  prefetch     batch_size
==============  ===========  ===========  ============
< 8 GB          min(2, cpu)  2            32
8–32 GB         min(4, cpu)  4            128
>= 32 GB        min(12, cpu) 8            256
==============  ===========  ===========  ============

``cpu`` is ``os.cpu_count() - 1`` (leaving one core for the main
process) clamped to at least 0.  When the cgroup or host memory
cannot be read the function falls back to the safest tier so a
misconfigured environment can never OOM through this code path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# cgroup v1 with no limit set returns a sentinel close to ``2 ** 63``;
# any value above this threshold is treated as "no limit".
_CGROUP_NO_LIMIT_THRESHOLD = 1 << 60

_FALLBACK_MEMORY_GB = 4.0  # used when neither cgroup nor sysconf works


@dataclass(frozen=True)
class DataLoaderSettings:
    """Resolved (workers, prefetch, batch_size) triple."""

    num_workers: int
    prefetch_factor: int
    batch_size: int


@dataclass(frozen=True)
class DataLoaderTune:
    """Snapshot of the inputs and outputs of one autodetect call.

    Serialised into ``manifest.json`` so a researcher can audit the
    DataLoader sizing decisions without having to re-derive them from
    the host they ran on.
    """

    cpu_count: int
    memory_budget_gb: float
    tier_name: str
    settings: DataLoaderSettings


@lru_cache(maxsize=1)
def autotune() -> DataLoaderTune:
    """Return the tier that matches the current host (cached).

    Pure read-only: no allocations, no GPU calls, safe to invoke at
    import time.  Logs the inputs (CPU count, memory budget) and the
    chosen tier at INFO **once per process**, subsequent calls return
    the cached result so step modules can ask for the settings as many
    times as they need without spamming the log.
    """
    cpu = max(1, (os.cpu_count() or 1))
    cpu_budget = max(1, cpu - 1)  # leave one core for the trainer
    mem_gb = _memory_budget_bytes() / (1024**3)

    if mem_gb < 8:
        settings = DataLoaderSettings(
            num_workers=min(2, cpu_budget),
            prefetch_factor=2,
            batch_size=32,
        )
        tier_name = "tight (<8 GB)"
    elif mem_gb < 32:
        settings = DataLoaderSettings(
            num_workers=min(4, cpu_budget),
            prefetch_factor=4,
            batch_size=128,
        )
        tier_name = "balanced (8-32 GB)"
    else:
        settings = DataLoaderSettings(
            num_workers=min(12, cpu_budget),
            prefetch_factor=8,
            batch_size=256,
        )
        tier_name = "loose (>=32 GB)"

    logger.info(
        "DataLoader autotune: cpu=%d mem=%.1fGB tier=%s -> "
        "num_workers=%d prefetch=%d batch_size=%d",
        cpu,
        mem_gb,
        tier_name,
        settings.num_workers,
        settings.prefetch_factor,
        settings.batch_size,
    )
    return DataLoaderTune(
        cpu_count=cpu,
        memory_budget_gb=round(mem_gb, 2),
        tier_name=tier_name,
        settings=settings,
    )


def autodetect() -> DataLoaderSettings:
    """Convenience accessor returning only the resolved settings."""
    return autotune().settings


def describe(config: dict | None = None) -> dict:
    """Return a JSON-serialisable snapshot of the autotune decision.

    Consumed by :mod:`src.utils.manifest` to embed the DataLoader
    sizing inputs and outputs in every run manifest.  When ``config``
    is provided and carries a ``dataloader:`` block, the resolved
    values reflect those overrides and the overridden keys appear
    under ``yaml_overrides`` so a researcher reading the manifest
    spots deliberate pinning at a glance.
    """
    tune = autotune()
    dl_cfg = (config or {}).get("dataloader") or {}
    overrides = {
        key: dl_cfg[key]
        for key in ("num_workers", "prefetch_factor", "batch_size")
        if dl_cfg.get(key) is not None
    }
    resolved = resolve_dataloader_settings(config)
    return {
        "cpu_count": tune.cpu_count,
        "memory_budget_gb": tune.memory_budget_gb,
        "tier": tune.tier_name,
        "auto": {
            "num_workers": tune.settings.num_workers,
            "prefetch_factor": tune.settings.prefetch_factor,
            "batch_size": tune.settings.batch_size,
        },
        "resolved": {
            "num_workers": resolved.num_workers,
            "prefetch_factor": resolved.prefetch_factor,
            "batch_size": resolved.batch_size,
        },
        "yaml_overrides": overrides,
    }


def resolve_dataloader_settings(config: dict | None = None) -> DataLoaderSettings:
    """Return the resolved settings: YAML overrides first, autotune as fallback.

    When ``config['dataloader']`` carries ``num_workers``,
    ``prefetch_factor`` or ``batch_size``, the YAML value wins.  Any
    field left unset (or set to ``None``) in the YAML falls through to
    the autotune tier.

    The whole ``dataloader:`` block is optional, so a config that does
    not mention it gets pure autotuned values.
    """
    auto = autodetect()
    dl_cfg = (config or {}).get("dataloader") or {}

    def _pick(key: str, fallback: int) -> int:
        value = dl_cfg.get(key)
        return fallback if value is None else int(value)

    return DataLoaderSettings(
        num_workers=_pick("num_workers", auto.num_workers),
        prefetch_factor=_pick("prefetch_factor", auto.prefetch_factor),
        batch_size=_pick("batch_size", auto.batch_size),
    )


def _memory_budget_bytes() -> int:
    """Return the strictest memory budget that applies to this process.

    Resolution order: cgroup v2 -> cgroup v1 -> host total memory ->
    a 4 GB fallback so the function never returns 0.
    """
    cgroup_v2 = _read_int_file(Path("/sys/fs/cgroup/memory.max"))
    if cgroup_v2 is not None and cgroup_v2 < _CGROUP_NO_LIMIT_THRESHOLD:
        return cgroup_v2

    cgroup_v1 = _read_int_file(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if cgroup_v1 is not None and cgroup_v1 < _CGROUP_NO_LIMIT_THRESHOLD:
        return cgroup_v1

    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass

    return int(_FALLBACK_MEMORY_GB * 1024**3)


def _read_int_file(path: Path) -> int | None:
    """Read *path* and parse it as an integer (cgroup interface convention).

    Returns ``None`` when the file does not exist, is unreadable, or
    contains a non-numeric value (e.g. cgroup v2 ``"max"``).
    """
    try:
        text = path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        return int(text)
    except ValueError:
        return None
