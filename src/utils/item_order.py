"""Canonical item order and its digest (feature/item identity, Q05).

``item2idx.json`` maps an external item id to its integer ``item_idx``.
Every feature matrix in the pipeline is positional — on-disk row ``i``
IS item ``i`` — so the order in which items are fed to an extractor must
be derived from the *mapped values*, never from the JSON/dict insertion
order (``{"b": 1, "a": 0}`` requires rows ``[a, b]``).  This module owns
that rule and the digest that lets a feature artifact prove which order
it was extracted in.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

#: Version of the ``item_order`` block persisted in feature sidecars.
ITEM_ORDER_SCHEMA_VERSION = 1


class ItemOrderError(ValueError):
    """``item2idx`` values are not exactly ``0..N-1`` (holes/duplicates/non-ints)."""


def canonical_item_order(item2idx: Mapping[str, object]) -> list[str]:
    """Return the item ids ordered by their mapped index (row ``i`` = index ``i``).

    :raises ItemOrderError: when the mapped values are not the integers
        ``0..N-1`` exactly once each — holes, duplicates or non-integer
        values all make a positional matrix undefined.
    """
    n_items = len(item2idx)
    order: list[str | None] = [None] * n_items
    duplicates: list[str] = []
    for item_id, value in item2idx.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ItemOrderError(f"item2idx[{item_id!r}] = {value!r} is not an integer index.")
        if not 0 <= value < n_items:
            raise ItemOrderError(
                f"item2idx[{item_id!r}] = {value} is outside [0, {n_items}): "
                "the mapping has a hole (indices must be exactly 0..N-1)."
            )
        if order[value] is not None:
            duplicates.append(f"{value} ({order[value]!r}, {item_id!r})")
            continue
        order[value] = str(item_id)
    if duplicates:
        raise ItemOrderError(
            f"item2idx maps {len(duplicates)} index(es) more than once, e.g. "
            f"{', '.join(duplicates[:5])}."
        )
    holes = [i for i, item_id in enumerate(order) if item_id is None]
    if holes:
        raise ItemOrderError(
            f"item2idx has {len(holes)} hole(s) in [0, {n_items}), e.g. {holes[:10]}."
        )
    return [item_id for item_id in order if item_id is not None]


def item_order_digest(item_ids: Sequence[object]) -> str:
    """SHA-256 over the ordered id list (ids stringified, canonical JSON)."""
    payload = json.dumps([str(item_id) for item_id in item_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def item_order_metadata(item_ids: Sequence[object]) -> dict:
    """The ``item_order`` block persisted next to a feature artifact."""
    return {
        "schema_version": ITEM_ORDER_SCHEMA_VERSION,
        "n_items": len(item_ids),
        "digest": item_order_digest(item_ids),
    }


def load_item_order(processed_dir: str | Path, dataset_name: str) -> list[str]:
    """Load ``<processed_dir>/<dataset>/item2idx.json`` in canonical order."""
    path = Path(processed_dir) / dataset_name / "item2idx.json"
    with open(path, encoding="utf-8") as fh:
        item2idx = json.load(fh)
    if not isinstance(item2idx, dict):
        raise ItemOrderError(f"{path}: expected a JSON object, got {type(item2idx).__name__}.")
    return canonical_item_order(item2idx)


__all__ = [
    "ITEM_ORDER_SCHEMA_VERSION",
    "ItemOrderError",
    "canonical_item_order",
    "item_order_digest",
    "item_order_metadata",
    "load_item_order",
]
