"""The ``resources:`` block (``configs/resources.yaml``), resolved and validated once.

Every computational limit of a run lives here: the share of the GPU a
process may claim, the host memory budget and what is withheld from
every pool, the worker counts of the training, fusion and DataLoader
pools, the DataLoader batch sizing, and how visual features are held.
All of it is *execution metadata*: it changes how fast and how much
memory a run uses, never what it computes, so none of it enters the
scientific identity (``src/utils/identity.py``); the run manifest
records the resolved block instead.

A missing file or block resolves to the conservative defaults declared
on each dataclass below (``configs/resources.yaml`` ships the
researcher's own values, e.g. ``gpu.vram_share: 0.95`` for unattended
nights against the desktop-friendly default of 0.5).  Every field is
validated at resolution: a negative, non-finite,
boolean, non-integer or out-of-range value, an unknown key at any
level, or one of the removed keys (``hp_search.workers``, a top-level
``dataloader:`` block, the flat M05-era ``resources.*`` names) raises
:class:`ValueError` naming the offending key and, for removed keys,
the key that replaces it.

``PRISM_VRAM_SHARE`` overrides ``resources.gpu.vram_share`` for one
launch (forwarded by ``docker-compose.yml``); it is read here, at
resolution, never at import time.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

VRAM_SHARE_ENV = "PRISM_VRAM_SHARE"
RESIDENCY_POLICIES = ("dense", "lazy", "auto")
AUTO = "auto"

_GIB = 1024**3

#: Flat M05-proposal keys and the nested key that replaced each of them.
_LEGACY_FLAT_KEYS = {
    "host_budget_bytes": "resources.host.budget_bytes",
    "headroom_bytes": "resources.host.headroom_bytes",
    "max_workers": "resources.workers.training",
    "feature_residency": "resources.features.residency",
}


@dataclass(frozen=True)
class GpuResources:
    """``vram_share``: fraction of the card one run may claim; ``ranking_vram_share``:
    share of that cap one ranking batch may use (shorter kernels keep the desktop alive)."""

    vram_share: float = 0.5
    ranking_vram_share: float = 0.125


@dataclass(frozen=True)
class HostResources:
    """``budget_bytes`` (``None`` = the cgroup limit), ``headroom_bytes`` withheld from
    every pool, ``reserved_bytes`` for the parent process and the page cache."""

    budget_bytes: int | None = None
    headroom_bytes: int = 4 * _GIB
    reserved_bytes: int = 4 * _GIB


@dataclass(frozen=True)
class WorkerResources:
    """Pool sizes: ``training`` (0 = auto-detect), ``fusion``, ``dataloader`` (``None`` = auto)."""

    training: int = 1
    fusion: int = 1
    dataloader: int | None = None


@dataclass(frozen=True)
class DataLoaderResources:
    """Pinned DataLoader sizing; ``None`` lets the autotune tier decide."""

    prefetch_factor: int | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class FeatureResources:
    """``residency`` (dense | lazy | auto) and ``item_block`` rows per catalogue-sized request."""

    residency: str = "dense"
    item_block: int = 8192


@dataclass(frozen=True)
class ResourcesConfig:
    """The resolved ``resources:`` block."""

    gpu: GpuResources = GpuResources()
    host: HostResources = HostResources()
    workers: WorkerResources = WorkerResources()
    dataloader: DataLoaderResources = DataLoaderResources()
    features: FeatureResources = FeatureResources()

    def to_payload(self) -> dict[str, Any]:
        """JSON-ready copy for the run manifest (``None`` shown as ``"auto"`` where it means that)."""
        payload = asdict(self)
        for section, key in (("workers", "dataloader"), ("dataloader", "prefetch_factor")):
            if payload[section][key] is None:
                payload[section][key] = AUTO
        if payload["dataloader"]["batch_size"] is None:
            payload["dataloader"]["batch_size"] = AUTO
        return payload


def resolve_resources(
    config: Mapping[str, Any] | None, *, env: Mapping[str, str] | None = None
) -> ResourcesConfig:
    """Validate and resolve the ``resources:`` block of *config* (defaults when absent).

    :param config: The merged framework configuration, or ``None``.
    :param env: Environment to read ``PRISM_VRAM_SHARE`` from (``os.environ`` by default).
    :returns: The resolved block.
    :raises ValueError: On any invalid, unknown or removed key (message names the key).
    """
    config = config or {}
    reject_removed_keys(config)
    block = config.get("resources") or {}
    if not isinstance(block, Mapping):
        raise ValueError(f"resources must be a mapping, got {type(block).__name__}")
    unknown = set(block) - {"gpu", "host", "workers", "dataloader", "features"}
    if unknown:
        raise ValueError(
            f"resources has unknown keys: {sorted(unknown)}; "
            "allowed: gpu, host, workers, dataloader, features"
        )
    gpu = _section(block, "gpu", ("vram_share", "ranking_vram_share"))
    host = _section(block, "host", ("budget_bytes", "headroom_bytes", "reserved_bytes"))
    workers = _section(block, "workers", ("training", "fusion", "dataloader"))
    loader = _section(block, "dataloader", ("prefetch_factor", "batch_size"))
    features = _section(block, "features", ("residency", "item_block"))
    vram_share = _env_vram_share(os.environ if env is None else env)
    if vram_share is None:
        vram_share = _fraction(gpu, "gpu", "vram_share", GpuResources.vram_share)
    return ResourcesConfig(
        gpu=GpuResources(
            vram_share=vram_share,
            ranking_vram_share=_fraction(
                gpu, "gpu", "ranking_vram_share", GpuResources.ranking_vram_share
            ),
        ),
        host=HostResources(
            budget_bytes=_count(host, "host", "budget_bytes", None, minimum=0, nullable=True),
            headroom_bytes=_count(host, "host", "headroom_bytes", 4 * _GIB, minimum=0),
            reserved_bytes=_count(host, "host", "reserved_bytes", 4 * _GIB, minimum=0),
        ),
        workers=WorkerResources(
            training=_count(workers, "workers", "training", 1, minimum=0),
            fusion=_count(workers, "workers", "fusion", 1, minimum=1),
            dataloader=_count(workers, "workers", "dataloader", None, minimum=0, auto=True),
        ),
        dataloader=DataLoaderResources(
            prefetch_factor=_count(
                loader, "dataloader", "prefetch_factor", None, minimum=1, auto=True
            ),
            batch_size=_count(loader, "dataloader", "batch_size", None, minimum=1, auto=True),
        ),
        features=FeatureResources(
            residency=_residency(features),
            item_block=_count(features, "features", "item_block", 8192, minimum=1),
        ),
    )


def reject_removed_keys(config: Mapping[str, Any]) -> None:
    """Fail on keys that ``configs/resources.yaml`` replaced, naming the new key.

    :raises ValueError: ``hp_search.workers``, a top-level ``dataloader:``
        block, or a flat M05-era ``resources.<key>``.
    """
    hp_search = config.get("hp_search")
    if isinstance(hp_search, Mapping) and "workers" in hp_search:
        raise ValueError(
            "hp_search.workers was removed: set resources.workers.training "
            "in configs/resources.yaml instead"
        )
    if "dataloader" in config:
        raise ValueError(
            "the top-level dataloader block was removed: set resources.workers.dataloader "
            "and resources.dataloader.{prefetch_factor,batch_size} in configs/resources.yaml"
        )
    block = config.get("resources")
    if not isinstance(block, Mapping):
        return
    for legacy, replacement in _LEGACY_FLAT_KEYS.items():
        if legacy in block:
            raise ValueError(f"resources.{legacy} was renamed: set {replacement} instead")


def _section(block: Mapping[str, Any], name: str, allowed: tuple[str, ...]) -> Mapping[str, Any]:
    sub = block.get(name)
    if sub is None:
        return {}
    if not isinstance(sub, Mapping):
        raise ValueError(f"resources.{name} must be a mapping, got {type(sub).__name__}")
    unknown = set(sub) - set(allowed)
    if unknown:
        raise ValueError(
            f"resources.{name} has unknown keys: {sorted(unknown)}; allowed: {list(allowed)}"
        )
    return sub


def _fraction(sub: Mapping[str, Any], section: str, key: str, default: float) -> float:
    value = sub.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"resources.{section}.{key} must be a number in (0, 1], got {value!r}")
    share = float(value)
    if not math.isfinite(share) or not 0.0 < share <= 1.0:
        raise ValueError(f"resources.{section}.{key} must be a number in (0, 1], got {value!r}")
    return share


def _count(
    sub: Mapping[str, Any],
    section: str,
    key: str,
    default: int | None,
    *,
    minimum: int,
    nullable: bool = False,
    auto: bool = False,
) -> int | None:
    value = sub.get(key, default)
    if value is None and (nullable or auto):
        return None
    if auto and value == AUTO:
        return None
    accepted = (
        f"an integer >= {minimum}"
        + (" or 'auto'" if auto else "")
        + (" or null" if nullable else "")
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"resources.{section}.{key} must be {accepted}, got {value!r}")
    return int(value)


def _residency(features: Mapping[str, Any]) -> str:
    value = features.get("residency", "dense")
    if value not in RESIDENCY_POLICIES:
        raise ValueError(
            f"resources.features.residency must be one of {RESIDENCY_POLICIES}, got {value!r}"
        )
    return str(value)


def _env_vram_share(env: Mapping[str, str]) -> float | None:
    raw = env.get(VRAM_SHARE_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        share = float(raw)
    except ValueError as exc:
        raise ValueError(f"{VRAM_SHARE_ENV}={raw!r} is not a number.") from exc
    if not math.isfinite(share) or not 0.0 < share <= 1.0:
        raise ValueError(f"{VRAM_SHARE_ENV}={raw!r} must be a fraction in (0, 1].")
    return share
