"""Auto-detected DataLoader settings for safe-by-default execution.

PyTorch's ``DataLoader`` exposes three knobs (``num_workers``,
``prefetch_factor``, ``batch_size``) that interact non-trivially with
the cgroup's CPU quota and memory budget.  Picking values
that fit *every* deployment the framework runs on, laptops, CI,
RunPod pods, lab servers, is impossible from a single hardcoded
default: too low wastes throughput on a 128 GB lab box; too high
gets the worker pool OOM-killed on a 16 GB laptop container.

This module replaces the hardcoded defaults with a small heuristic
that inspects the runtime environment once at startup and picks a
tier that fits.  A researcher who wants an exact value pins it in
``configs/resources.yaml`` (``resources.workers.dataloader`` and
``resources.dataloader.{prefetch_factor,batch_size}``); a pinned value
wins over the tier, ``auto`` falls through to it.

Tiers (memory budget refers to the cgroup limit when running in a
container, the total host RAM otherwise):

==============  ===========  ===========  ============
memory budget   num_workers  prefetch     batch_size
==============  ===========  ===========  ============
< 8 GB          min(2, cpu)  2            32
8–32 GB         min(4, cpu)  4            128
>= 32 GB        min(12, cpu) 8            256
==============  ===========  ===========  ============

``cpu`` is ``available_cpus() - 1`` (leaving one core for the main
process) clamped to at least 0.  When the cgroup or host memory
cannot be read the function falls back to the safest tier so a
misconfigured environment can never OOM through this code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.utils.logging import get_logger

# The budget itself is shared with the process-pool sizing in
# :mod:`src.utils.memory`; re-exported under the private name this
# module has always used so callers (and tests) keep one entry point.
from src.utils.memory import available_cpus
from src.utils.memory import memory_budget_bytes as _memory_budget_bytes
from src.utils.resources import ResourcesConfig, resolve_resources

logger = get_logger(__name__)


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
    cpu = available_cpus()
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
    sizing inputs and outputs in every run manifest.  The resolved
    values reflect the pins in ``resources.workers.dataloader`` /
    ``resources.dataloader``, and the pinned keys appear under
    ``yaml_overrides`` so a researcher reading the manifest spots
    deliberate pinning at a glance.
    """
    tune = autotune()
    pinned = _pinned(resolve_resources(config))
    overrides = {key: value for key, value in pinned.items() if value is not None}
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


def _pinned(resources: ResourcesConfig) -> dict[str, int | None]:
    """The three DataLoader pins of the resolved block (``None`` = auto)."""
    return {
        "num_workers": resources.workers.dataloader,
        "prefetch_factor": resources.dataloader.prefetch_factor,
        "batch_size": resources.dataloader.batch_size,
    }


def resolve_dataloader_settings(config: dict | None = None) -> DataLoaderSettings:
    """Return the resolved settings: ``resources`` pins first, autotune as fallback.

    ``resources.workers.dataloader``, ``resources.dataloader.prefetch_factor``
    and ``resources.dataloader.batch_size`` win when pinned to an
    integer; ``auto`` (or an absent block) falls through to the autotune
    tier.  ``num_workers`` is always clamped by the CPU quota.

    :param config: The merged framework configuration, or ``None`` for pure autotune.
    :returns: The settings every DataLoader in the pipeline is built with.
    :raises ValueError: On an invalid ``resources`` block or a removed
        top-level ``dataloader:`` block.
    """
    auto = autodetect()
    pinned = _pinned(resolve_resources(config))

    def _pick(key: str, fallback: int) -> int:
        value = pinned[key]
        return fallback if value is None else int(value)

    # A pinned num_workers states intent, not a licence to oversubscribe:
    # loader processes above the cgroup's CPU quota only add context
    # switching, and on a workstation that contention is felt as a frozen
    # desktop.  One core is left for the process doing the training.
    cpu_cap = max(1, available_cpus() - 1)
    requested = _pick("num_workers", auto.num_workers)
    num_workers = min(requested, cpu_cap)
    if num_workers < requested:
        logger.info(
            "DataLoader num_workers clamped %d -> %d by the CPU quota (%d cores).",
            requested,
            num_workers,
            available_cpus(),
        )

    return DataLoaderSettings(
        num_workers=num_workers,
        prefetch_factor=_pick("prefetch_factor", auto.prefetch_factor),
        batch_size=_pick("batch_size", auto.batch_size),
    )
