"""Pluggable dataset providers and the on-disk schema they must produce.

A *dataset provider* is the glue between a raw dataset (whatever its
original shape) and the canonical layout that the rest of the pipeline
expects.  Once a dataset is materialised in this layout, every step
from ``extract`` onward (extract → finetune → fuse → train → evaluate
→ statistical → export_best) is dataset-agnostic.

Canonical layout
================

Given a logical dataset name ``<name>`` (e.g. ``"amazon_fashion"``),
the provider must produce the following files / directories::

    data/raw/<name>/                          # provider-specific raw data
    data/raw/<name>/images/<item_idx>.jpg     # one image per item
    data/processed/<name>/train.csv           # user_idx,item_idx (header included)
    data/processed/<name>/val.csv             # idem (1 row per user, leave-one-out)
    data/processed/<name>/test.csv            # idem (1 row per user, leave-one-out)
    data/processed/<name>/user2idx.json       # {"<external user id>": <user_idx>}
    data/processed/<name>/item2idx.json       # {"<external item id>": <item_idx>}

Schema invariants
-----------------

The pipeline relies on the following invariants — break them and you
will see ``IndexError``\\s in the recommender models or silently wrong
metrics.

* ``user_idx`` ranges over ``[0, n_users)`` *contiguously*.
* ``item_idx`` ranges over ``[0, n_items)`` *contiguously*.
* ``user2idx.json`` and ``item2idx.json`` have ``len == n_users`` /
  ``n_items`` and map original (string) ids to integer indices.
* ``train.csv`` / ``val.csv`` / ``test.csv`` have exactly two columns
  named ``user_idx`` and ``item_idx`` (header included).
* ``val`` and ``test`` follow a leave-one-out protocol — one held-out
  ``item_idx`` per user.  Multiple rows per user are tolerated by the
  evaluator but are not part of the canonical evaluation protocol
  defined by this framework.
* Image filenames are ``<item_idx>.jpg`` (or ``.jpeg``/``.png``/
  ``.webp``).  Items without images are dropped during extraction.

Adding a new dataset
====================

Subclass :class:`DatasetProvider`, implement the three required hooks,
and register the subclass under one or more dataset names::

    from pathlib import Path
    import pandas as pd

    from src.data.base import (
        DatasetProvider,
        register_dataset_provider,
        write_processed_splits,
    )

    class MyDataset(DatasetProvider):
        name = "my_dataset"

        def download(self) -> None:
            # populate self.raw_dir however you like
            ...

        def save_processed(self, processed_dir) -> None:
            # processed_dir is already the final per-dataset directory
            # (the caller appends the dataset name) — do NOT append
            # self.name again.
            train = pd.DataFrame({"user_idx": [...], "item_idx": [...]})
            val   = pd.DataFrame({"user_idx": [...], "item_idx": [...]})
            test  = pd.DataFrame({"user_idx": [...], "item_idx": [...]})
            user2idx = {"u1": 0, "u2": 1, ...}
            item2idx = {"i1": 0, "i2": 1, ...}
            write_processed_splits(
                processed_dir,
                train, val, test, user2idx, item2idx,
            )

        def extract_images(self, image_dir) -> None:
            # write <image_dir>/<item_idx>.jpg for every item
            ...

    register_dataset_provider("my_dataset", MyDataset)

Then add ``"my_dataset"`` to the ``datasets:`` list in
``configs/default.yaml`` and run the pipeline as usual.

Optional: implement :meth:`load_categories` / :meth:`num_categories`
to enable the fine-tuning step on the new dataset.  Datasets without
categories are skipped during fine-tuning and can borrow weights from
another dataset via ``finetuning.tradesy_transfer_from`` in
``configs/finetuning.yaml``.

A complete, runnable example lives at :mod:`src.data.example_csv`.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetProvider(abc.ABC):
    """Adapter that materialises a dataset in the pipeline's canonical layout."""

    #: Logical dataset name (also used as the directory name on disk).
    name: str = ""

    def __init__(self, name: str = "", raw_dir: str | Path | None = None) -> None:
        if name:
            self.name = name
        if not self.name:
            raise ValueError(
                f"{type(self).__name__} requires a non-empty dataset name "
                f"(set the class attribute or pass it to __init__)."
            )
        base = Path(raw_dir) if raw_dir else Path("data/raw")
        self.raw_dir = base / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def download(self) -> None:
        """Materialise the raw dataset under ``self.raw_dir``.

        Implementations must be idempotent — running ``download()``
        twice on the same machine should leave the second invocation
        as a no-op once the data is on disk.  When possible, validate
        the size / checksum of already-present files so a previous
        truncated download is detected and re-fetched.
        """

    @abc.abstractmethod
    def save_processed(self, processed_dir: str | Path) -> None:
        """Write ``train.csv`` / ``val.csv`` / ``test.csv`` and the
        ``user2idx.json`` / ``item2idx.json`` mappings.

        ``processed_dir`` is the **final** per-dataset directory: the
        caller (``src.steps.preprocess``) already appends the dataset
        name, so implementations write directly into it and must NOT
        append ``self.name`` again.

        The :func:`write_processed_splits` helper takes care of the
        boilerplate (column names, JSON formatting, ID validation) and
        is the recommended way to implement this hook.
        """

    @abc.abstractmethod
    def extract_images(self, image_dir: str | Path) -> None:
        """Write per-item images under ``image_dir``.

        File name convention: ``<item_idx>.jpg`` (``.jpeg``/``.png``/
        ``.webp`` are also recognised).  Items without an image may be
        omitted; the rest of the pipeline gracefully drops them.
        """

    def load_categories(self) -> dict[str, int] | None:
        """Return ``{external_item_id (str): category_label (int)}`` or
        ``None`` if absent.

        The keys are the *external* item ids — the same string that
        names the per-item image file on disk (e.g. ``"0"`` for DVBPR,
        ``"i_42"`` for arbitrary CSV-based datasets).  This contract is
        what :class:`src.finetuning.dataset.CategoryDataset` consumes.

        Datasets without categories are still fully usable by the
        pipeline; they are simply skipped during fine-tuning unless
        another fine-tuned dataset's weights are configured as the
        transfer source.
        """
        return None

    def num_categories(self) -> int:
        """Return the number of category classes (``0`` when absent)."""
        return 0


def write_processed_splits(
    output_dir: str | Path,
    train: pd.DataFrame | list[tuple[int, int]],
    val: pd.DataFrame | list[tuple[int, int]],
    test: pd.DataFrame | list[tuple[int, int]],
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> None:
    """Persist the three split CSVs + the two JSON mappings atomically.

    Parameters
    ----------
    output_dir:
        Destination directory.  Will be created if necessary.  This is
        the final per-dataset directory (the ``processed_dir`` passed to
        ``save_processed``), not ``processed_dir / self.name``.
    train, val, test:
        Either a :class:`pandas.DataFrame` with columns
        ``user_idx, item_idx`` or an iterable of ``(user_idx, item_idx)``
        tuples.
    user2idx, item2idx:
        Mappings from the original (string) IDs to the contiguous
        integer indices used internally.  Their values must cover
        ``[0, len(mapping))`` exactly once each.

    Raises
    ------
    ValueError:
        When an ID mapping is not contiguous, or when a CSV references
        an index outside the mapping range.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_users = len(user2idx)
    n_items = len(item2idx)
    _check_contiguous(user2idx.values(), n_users, "user2idx")
    _check_contiguous(item2idx.values(), n_items, "item2idx")

    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        df = _coerce_to_dataframe(split, split_name)
        _check_indices(df, n_users, n_items, split_name)
        df.to_csv(output_dir / f"{split_name}.csv", index=False)

    with open(output_dir / "user2idx.json", "w", encoding="utf-8") as fh:
        json.dump(user2idx, fh)
    with open(output_dir / "item2idx.json", "w", encoding="utf-8") as fh:
        json.dump(item2idx, fh)

    logger.info(
        "Wrote processed splits to %s (users=%d, items=%d)",
        output_dir,
        n_users,
        n_items,
    )


def _coerce_to_dataframe(split, split_name: str) -> pd.DataFrame:
    if isinstance(split, pd.DataFrame):
        if list(split.columns) != ["user_idx", "item_idx"]:
            raise ValueError(
                f"{split_name}.csv must have columns ['user_idx', 'item_idx']; "
                f"got {list(split.columns)}"
            )
        return split.astype({"user_idx": int, "item_idx": int})
    return pd.DataFrame(list(split), columns=["user_idx", "item_idx"]).astype(int)


def _check_contiguous(values, expected: int, label: str) -> None:
    seen = sorted(int(v) for v in values)
    if seen != list(range(expected)):
        raise ValueError(
            f"{label} indices must be contiguous in [0, {expected}); "
            f"got {len(seen)} entries with min={seen[0] if seen else 'n/a'} "
            f"max={seen[-1] if seen else 'n/a'}"
        )


def _check_indices(df: pd.DataFrame, n_users: int, n_items: int, split_name: str) -> None:
    if df.empty:
        return
    if df["user_idx"].min() < 0 or df["user_idx"].max() >= n_users:
        raise ValueError(f"{split_name}.csv has user_idx out of range [0, {n_users})")
    if df["item_idx"].min() < 0 or df["item_idx"].max() >= n_items:
        raise ValueError(f"{split_name}.csv has item_idx out of range [0, {n_items})")


def validate_layout(
    name: str,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    *,
    sample_image_count: int = 5,
) -> list[str]:
    """Sanity-check the on-disk layout for ``name``.

    Returns the list of detected problems (empty list = layout OK).
    Useful both as a self-test for new providers and as a smoke test
    before launching a multi-day grid search.

    The optional *sample_image_count* parameter controls how many
    random item ids are drawn from ``item2idx.json`` and verified
    on disk; pass ``0`` to skip the spot-check (useful in CI where
    images are not committed).
    """
    problems: list[str] = []
    raw = Path(raw_dir) / name
    proc = Path(processed_dir) / name

    if not raw.exists():
        problems.append(f"missing raw directory: {raw}")
    images_dir = raw / "images"
    if not images_dir.exists():
        problems.append(f"missing image directory: {images_dir}")
    if not proc.exists():
        problems.append(f"missing processed directory: {proc}")

    for fname in ("train.csv", "val.csv", "test.csv"):
        if not (proc / fname).exists():
            problems.append(f"missing {proc}/{fname}")

    for fname in ("user2idx.json", "item2idx.json"):
        if not (proc / fname).exists():
            problems.append(f"missing {proc}/{fname}")

    if problems:
        return problems

    try:
        with open(proc / "item2idx.json") as fh:
            item2idx = json.load(fh)
        n_items = len(item2idx)
        if sorted(int(v) for v in item2idx.values()) != list(range(n_items)):
            problems.append(f"item2idx values not contiguous in [0, {n_items})")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not parse item2idx.json: {exc}")
        return problems

    try:
        with open(proc / "user2idx.json") as fh:
            user2idx = json.load(fh)
        n_users = len(user2idx)
        if sorted(int(v) for v in user2idx.values()) != list(range(n_users)):
            problems.append(f"user2idx values not contiguous in [0, {n_users})")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not parse user2idx.json: {exc}")

    if images_dir.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files_by_stem = {
            entry.stem
            for entry in images_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() in valid_exts
        }
        n_files = len(files_by_stem)
        if n_files < n_items // 2:
            problems.append(
                f"only {n_files} images found in {images_dir} for "
                f"{n_items} items — coverage below 50%"
            )

        if sample_image_count > 0 and n_items > 0:
            import random

            rng = random.Random(0)
            sample = rng.sample(list(item2idx.keys()), min(sample_image_count, n_items))
            missing = [iid for iid in sample if str(iid) not in files_by_stem]
            if missing:
                problems.append(
                    f"missing images for sampled items "
                    f"(checked {len(sample)}, missing {len(missing)}): "
                    f"{missing[:3]}..."
                )

    return problems


_REGISTRY: dict[str, Callable[[], DatasetProvider]] = {}


def register_dataset_provider(
    name: str,
    factory: Callable[[], DatasetProvider],
) -> None:
    """Register ``factory`` under ``name`` so the pipeline can resolve it later."""
    _REGISTRY[name] = factory


def get_dataset_provider(name: str) -> DatasetProvider:
    """Return a fresh provider instance for ``name``.

    Raises :class:`KeyError` (with a helpful message listing the
    available providers) when ``name`` has not been registered.
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        available = sorted(_REGISTRY)
        raise KeyError(
            f"No DatasetProvider registered for {name!r}.  "
            f"Available providers: {available}.  "
            f"Register a custom provider via "
            f"src.data.base.register_dataset_provider(name, factory)."
        )
    return factory()


def registered_dataset_names() -> list[str]:
    """Return the sorted list of currently-registered dataset names."""
    return sorted(_REGISTRY)
