"""Derive item categories from a McAuley-style textual taxonomy.

Many DVBPR-derived splits (e.g. ``amazon_men``/``amazon_women``/
``tradesy``) ship a list of taxonomy paths per item but not the one-hot
``c`` vector the fine-tuning step expects.  This module turns that
taxonomy into a ``{item_id: contiguous_label}`` mapping the framework
can consume via the standard ``data/raw/<name>/categories.csv``
sidecar.

The functions here are pure (no IO) so they are easy to test and
reuse from any provider that lands McAuley-style metadata.  Providers
that need to persist the result can call :func:`write_categories_csv`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.utils.atomic_io import atomic_write
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _decode(value: Any) -> str:
    """Decode ``bytes`` to ``str``; leave ``str`` unchanged."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin1", errors="replace")
    return str(value)


def extract_taxonomy(item: Any) -> list[str] | None:
    """Pull the ``categories`` taxonomy from a single item record.

    Items in DVBPR ``.npy`` files are dicts whose keys may be bytes or
    strings, and whose ``categories`` value is a list of taxonomy paths.
    This helper returns the first non-empty path as a list of textual
    nodes (decoded), or ``None`` when no usable path is present.
    """
    if not isinstance(item, dict):
        return None
    raw = item.get("categories")
    if raw is None:
        raw = item.get(b"categories")
    if raw is None:
        return None
    try:
        first_path = next((p for p in raw if p), None)
    except TypeError:
        return None
    if first_path is None:
        return None
    return [_decode(node) for node in first_path]


def _label_for(taxonomy: list[str], level: int) -> str | None:
    """Pick the requested taxonomy depth, falling back to the deepest one."""
    if not taxonomy:
        return None
    if level >= len(taxonomy):
        return taxonomy[-1]
    return taxonomy[level]


def derive_categories(
    items: Iterable[tuple[Any, Any]],
    *,
    level: int = 2,
    min_samples: int = 50,
) -> dict[str, int] | None:
    """Build ``{item_id: contiguous_label}`` from a McAuley-style taxonomy.

    Parameters
    ----------
    items:
        Iterable of ``(item_id, item_record)`` pairs.  ``item_record``
        is whatever the provider has on hand (typically a dict with a
        ``categories`` field).
    level:
        Taxonomy depth to use as the label (0 = root, increases toward
        the leaves).  Levels deeper than an item's taxonomy fall back
        to that item's deepest available level.  Default ``2`` matches
        the broad-category granularity that works for DVBPR fashion.
    min_samples:
        Drop labels with fewer than ``min_samples`` items.  Items in
        dropped labels are excluded from the returned mapping so they
        do not pollute fine-tuning.

    Returns
    -------
    ``{str item_id: int label}`` with labels remapped to a contiguous
    ``[0, n_classes)`` range, or ``None`` if no item has a usable
    taxonomy.
    """
    raw_pairs: list[tuple[str, str]] = []
    skipped_no_taxonomy = 0
    for item_id, record in items:
        taxonomy = extract_taxonomy(record)
        label = _label_for(taxonomy, level) if taxonomy else None
        if label is None:
            skipped_no_taxonomy += 1
            continue
        raw_pairs.append((str(item_id), label))

    if not raw_pairs:
        return None

    counts = Counter(label for _, label in raw_pairs)
    kept_labels = {lbl for lbl, n in counts.items() if n >= min_samples}
    if not kept_labels:
        return None

    label_remap = {lbl: idx for idx, lbl in enumerate(sorted(kept_labels))}
    mapping = {item_id: label_remap[label] for item_id, label in raw_pairs if label in kept_labels}

    logger.info(
        "Derived %d categories from taxonomy level %d (kept %d/%d labels, "
        "min_samples=%d, %d items without taxonomy)",
        len(kept_labels),
        level,
        len(kept_labels),
        len(counts),
        min_samples,
        skipped_no_taxonomy,
    )
    return mapping


def write_categories_csv(mapping: dict[str, int], path: str | Path) -> None:
    """Persist ``{item_id: label}`` as ``item_id,category_label`` CSV.

    Writes through a temp file + atomic rename so a crash mid-write
    cannot leave a partial CSV that the rest of the pipeline would
    silently misread.
    """
    target = Path(path)
    lines = ["item_id,category_label\n"]
    lines.extend(f"{item_id},{label}\n" for item_id, label in mapping.items())
    payload = "".join(lines)
    atomic_write(lambda tmp: Path(tmp).write_text(payload, encoding="utf-8"), target)
    logger.info("Wrote %d category rows to %s", len(mapping), target)


class CategoryContractError(RuntimeError):
    """Raised when a dataset's loaded categories violate its declared contract.

    See :func:`enforce_category_contract` and the ``dataset_contracts``
    block in ``configs/default.yaml``.
    """


def enforce_category_contract(
    dataset_name: str,
    expects_categories: bool,
    categories: dict[str, int] | None,
) -> None:
    """Fail loud when loaded categories disagree with the declared contract.

    ``categories`` is the value returned by
    :meth:`DatasetProvider.load_categories` — the exact silent signal
    that otherwise decides whether DeepStyle degenerates into VBPR and
    whether fine-tuning falls back to transfer weights.  This function
    turns a mismatch between that signal and the ``expects_categories``
    declaration into an explicit error instead of a silent behaviour flip.

    Parameters
    ----------
    dataset_name:
        Name of the dataset being checked (for the error message).
    expects_categories:
        The contract declared in ``configs/default.yaml`` under
        ``dataset_contracts.<name>.expects_categories``.
    categories:
        The ``{item_id: label}`` mapping (or ``None``/empty) the provider
        actually loaded.

    Raises
    ------
    CategoryContractError:
        When the declaration and the loaded data disagree in either
        direction.
    """
    n_categories = len(set(categories.values())) if categories else 0
    has_categories = n_categories > 0

    if expects_categories and not has_categories:
        raise CategoryContractError(
            f"{dataset_name!r} declares expects_categories=true in "
            f"configs/default.yaml, but the provider loaded no category "
            f"labels (load_categories() returned "
            f"{'an empty mapping' if categories is not None else 'None'}). "
            f"Either the raw data is missing its category taxonomy or the "
            f"contract is wrong — resolve before running the pipeline."
        )
    if not expects_categories and has_categories:
        raise CategoryContractError(
            f"{dataset_name!r} declares expects_categories=false in "
            f"configs/default.yaml, but the provider loaded "
            f"{len(categories)} labelled items across {n_categories} "
            f"categories. This would silently stop DeepStyle from "
            f"degenerating into VBPR and disable fine-tuning transfer. "
            f"Either the raw data unexpectedly ships a taxonomy or the "
            f"contract is wrong — resolve before running the pipeline."
        )


def item_category_array(dataset_name: str, processed_dir: str | Path):
    """Build the ``(n_items,)`` category-index array for a dataset.

    Reuses the same category labels the fine-tuning step consumes
    (``DatasetProvider.load_categories``) and the canonical
    ``item2idx.json`` mapping — built ONCE before training, so the
    recommender does a plain batched tensor lookup at train time.

    Returns ``None`` when the dataset ships no category labels (e.g.
    Tradesy): the caller (DeepStyle) then uses a single null category,
    which analytically degenerates the model to VBPR — an expected,
    declared property, not an error.

    Items present in the catalogue but absent from the labels get a
    dedicated extra index (``n_categories``), a learned "unlabelled"
    bucket, so no item silently shares a real category.
    """
    import json

    import numpy as np

    from src.data.base import get_dataset_provider

    provider = get_dataset_provider(dataset_name)
    categories = provider.load_categories()
    if not categories:
        return None

    with open(Path(processed_dir) / dataset_name / "item2idx.json", encoding="utf-8") as fh:
        item2idx = json.load(fh)

    n_items = len(item2idx)
    n_categories = max(categories.values()) + 1
    arr = np.full(n_items, n_categories, dtype=np.int64)  # default: unlabelled bucket
    for external_id, idx in item2idx.items():
        label = categories.get(str(external_id))
        if label is not None:
            arr[int(idx)] = int(label)
    n_unlabelled = int((arr == n_categories).sum())
    if n_unlabelled:
        logger.info(
            "%s: %d/%d items without a category label -> learned 'unlabelled' bucket.",
            dataset_name,
            n_unlabelled,
            n_items,
        )
    return arr
