"""Uniform labels for per-cell cost entries.

Every cost cell in ``step_timings.json`` / ``cell_costs.csv`` names the
dimensions a researcher groups by: ``dataset``, ``extractors``,
``fusion`` and — where one exists — ``model``.  Steps know those
dimensions under different shapes (an embedding stem in ``train``, a
list of source paths in ``fuse``), so this module resolves them once,
in one place, and every step labels its cells the same way.

A hybrid stem does not carry its backbones in its name
(``hybrid_max_pool_learned_D128``), so they are read back from the
artifact's provenance sidecar, recursing through projected or
re-fused sources down to the leaf backbones.  The resolution is
best-effort: a label is metadata, and a missing sidecar must never
fail the cell it describes.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.artifact_names import FUSION_PREFIX
from src.utils.identity import read_provenance
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Stem of the pure collaborative baseline, which consumes no features.
NO_EMBEDDING = "none"

_ARTIFACT_SUFFIXES = (".npy", ".json")


def _known_extractors() -> list[str]:
    """Registered backbone names, longest first (names prefix each other)."""
    from src.extractors import registered_extractor_names

    return sorted(registered_extractor_names(), key=len, reverse=True)


def extractor_of(stem: str) -> str | None:
    """Backbone a native stem was produced by, or ``None``.

    Longest-match against the registered names, so ``cvt_13_pcaw128``
    and ``resnet50_finetuned`` resolve to ``cvt_13`` and ``resnet50``.
    """
    for name in _known_extractors():
        if stem == name or stem.startswith(f"{name}_"):
            return name
    return None


def _stem(name: str) -> str:
    for suffix in _ARTIFACT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _unique(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _source_names(payload: dict[str, Any]) -> list[str]:
    """Artifact names a provenance record was derived from, in order."""
    if payload.get("kind") == "projection":
        source = payload.get("source") or {}
        return [str(source["name"])] if source.get("name") else []
    return [str(s["name"]) for s in payload.get("sources") or [] if s.get("name")]


def _read(artifact: Path) -> dict[str, Any] | None:
    try:
        return read_provenance(artifact)
    except Exception as exc:  # noqa: BLE001 - a label must never fail its cell
        logger.debug("unreadable provenance for %s: %r", artifact, exc)
        return None


@lru_cache(maxsize=4096)
def _hybrid_labels(artifact: str, depth: int = 0) -> tuple[tuple[str, ...], str | None]:
    """``(backbones, strategy)`` behind *artifact*, recursing through sidecars.

    A projection names one source and a fusion names several; either may
    itself be a hybrid, so the walk follows sibling artifacts until it
    reaches native backbones.  The strategy is the outermost fusion's.
    """
    path = Path(artifact)
    payload = _read(path) if depth < 8 else None
    if not payload:
        base = extractor_of(_stem(path.name))
        return ((base,) if base else ()), None
    extractors: list[str] = []
    strategy = payload.get("strategy")
    for name in _source_names(payload):
        inner, inner_strategy = _hybrid_labels(str(path.with_name(name)), depth + 1)
        extractors.extend(inner)
        strategy = strategy or inner_strategy
    return tuple(_unique(extractors)), strategy


def embedding_labels(embedding: str, artifact: str | Path | None = None) -> dict[str, Any]:
    """Resolve the ``embedding`` / ``extractors`` / ``fusion`` labels of a stem.

    :param embedding: The embedding stem a cell consumes (``none``,
        ``resnet50``, ``clip_vitb32_pcaw128``, ``hybrid_concat``...).
    :param artifact: Path of the feature artifact behind the stem.  Only
        consulted for ``hybrid_*`` stems, whose backbones live in the
        provenance sidecar rather than in the name.
    :returns: A dict with ``embedding``, ``extractors`` (list, empty for
        the baseline) and ``fusion`` (strategy name or ``None``).
    """
    labels: dict[str, Any] = {"embedding": embedding, "extractors": [], "fusion": None}
    if embedding == NO_EMBEDDING:
        return labels
    if not embedding.startswith(FUSION_PREFIX) or artifact is None:
        base = extractor_of(embedding)
        labels["extractors"] = [base] if base else []
        return labels
    extractors, strategy = _hybrid_labels(str(artifact))
    labels["extractors"] = list(extractors)
    labels["fusion"] = strategy
    return labels


def source_labels(source_paths: Iterable[str | Path]) -> list[str]:
    """Backbones behind a list of fusion source artifacts, in order.

    :param source_paths: The artifacts a fusion consumes; native,
        projected or hybrid alike.
    :returns: The distinct leaf backbones, in first-seen order.
    """
    found: list[str] = []
    for path in source_paths:
        found.extend(_hybrid_labels(str(path))[0])
    return _unique(found)
