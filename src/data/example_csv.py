"""Worked example: a CSV-driven dataset provider.

This module is **not** registered automatically — it is a runnable
template you can copy and adapt.  The provider expects the user to
already have:

* a CSV with columns ``user_id, item_id, image_path`` describing every
  observed interaction (one row per click/like/purchase);
* a directory of images.

The provider builds the canonical layout described in
:mod:`src.data.base` from that CSV using a simple leave-one-out split.

Usage
-----

::

    from pathlib import Path
    from src.data.base import register_dataset_provider
    from src.data.example_csv import CSVDatasetProvider

    def my_factory() -> CSVDatasetProvider:
        return CSVDatasetProvider(
            name="my_dataset",
            interactions_csv=Path("/path/to/interactions.csv"),
            images_dir=Path("/path/to/images/"),
        )

    register_dataset_provider("my_dataset", my_factory)

After this, ``"my_dataset"`` can be added to the ``datasets:`` list in
``configs/default.yaml`` and the rest of the pipeline picks it up
without any code changes.

CSV schema
----------

``interactions.csv`` (required)::

    user_id,item_id,image_path
    u_1,it_42,images/it_42.jpg
    u_1,it_91,images/it_91.jpg
    u_2,it_42,images/it_42.jpg
    ...

* ``user_id`` and ``item_id`` are arbitrary strings.  The provider
  assigns the integer indices required by the pipeline (contiguous in
  ``[0, n_users)`` / ``[0, n_items)``).
* ``image_path`` is resolved relative to ``images_dir``.
* Every ``(user_id, item_id)`` row counts as a positive interaction.

``categories.csv`` (optional — enables fine-tuning)::

    item_id,category_label
    it_42,3
    it_91,1
    ...

* When present, the provider exposes :meth:`load_categories` /
  :meth:`num_categories`, which makes the dataset eligible for the
  fine-tuning step (Battery 2).
* ``category_label`` may be any integer — the provider remaps the
  labels to a contiguous ``[0, n_classes)`` range internally.

Splitting
---------

The default split is leave-one-out: each user contributes 1 random
interaction to ``test``, 1 to ``val``, and the rest to ``train``.
Users with fewer than 3 interactions are dropped.  Override
:meth:`CSVDatasetProvider.split` to use a different protocol.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.base import (
    DatasetProvider,
    write_processed_splits,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CSVDatasetProvider(DatasetProvider):
    """Builds the canonical pipeline layout from a (user, item, image) CSV."""

    def __init__(
        self,
        name: str,
        interactions_csv: str | Path,
        images_dir: str | Path,
        raw_dir: str | Path | None = None,
        seed: int = 42,
        min_user_interactions: int = 3,
        categories_csv: str | Path | None = None,
    ) -> None:
        super().__init__(name=name, raw_dir=raw_dir)
        self.interactions_csv = Path(interactions_csv)
        self.images_dir = Path(images_dir)
        self.seed = seed
        self.min_user_interactions = min_user_interactions
        self.categories_csv = Path(categories_csv) if categories_csv else None

        self._df: pd.DataFrame | None = None
        self._user2idx: dict[str, int] | None = None
        self._item2idx: dict[str, int] | None = None
        self._item_image_paths: dict[int, Path] | None = None
        self._categories: dict[str, int] | None = None
        self._n_categories: int = 0
        self._categories_loaded: bool = False

    def download(self) -> None:
        """Copy / link the source CSV into ``raw_dir`` (idempotent)."""
        target = self.raw_dir / "interactions.csv"
        if target.exists():
            logger.info("%s: interactions.csv already in place, skipping.", self.name)
            return

        if not self.interactions_csv.exists():
            raise FileNotFoundError(
                f"CSVDatasetProvider({self.name!r}): "
                f"interactions_csv not found at {self.interactions_csv}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.interactions_csv, target)
        logger.info("%s: copied interactions.csv to %s", self.name, target)

    def save_processed(self, processed_dir: str | Path) -> None:
        """Write the canonical train/val/test CSV layout into *processed_dir*.

        The contract matches :meth:`DVBPRDataLoader.save_processed`:
        callers pass the **full** target directory
        (``data/processed/<name>/``); this method does not append its
        own ``self.name`` again.
        """
        df, user2idx, item2idx = self._loaded()
        train, val, test = self.split(df, user2idx, item2idx)
        write_processed_splits(
            Path(processed_dir),
            train, val, test,
            user2idx, item2idx,
        )

    def extract_images(self, image_dir: str | Path) -> None:
        """Materialise per-item images at ``image_dir`` named by *external* id.

        ``ImageDataset`` (in ``src/steps/extract.py``) matches files by
        stem against the *keys* of ``item2idx.json`` — i.e. the external
        item id supplied in ``interactions.csv``.  We therefore name
        each output file with that same external id so the downstream
        lookup succeeds for any custom dataset.

        The ``image_path`` column is interpreted *relative to the
        dataset directory* (the parent of ``images/``), matching the
        contract documented in ``datasets/plugins/_example/`` —
        ``image_path: images/foo.jpg`` resolves to
        ``<dataset_dir>/images/foo.jpg``.  As a fallback, paths
        already relative to ``images_dir`` (legacy CSVs) are also
        accepted.
        """
        df, _, item2idx = self._loaded()
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = self.images_dir.parent

        item_images: dict[str, Path] = {}
        for item_str in item2idx:
            row = df.loc[df["item_id"] == item_str].iloc[0]
            rel = str(row["image_path"])
            src = (dataset_dir / rel).resolve()
            if not src.exists():
                src = (self.images_dir / rel).resolve()
            if src.exists():
                item_images[item_str] = src

        copied = 0
        skipped = 0
        for item_str, src in item_images.items():
            ext = src.suffix.lower() or ".jpg"
            dest = image_dir / f"{item_str}{ext}"
            if dest.exists():
                skipped += 1
                continue
            shutil.copyfile(src, dest)
            copied += 1
        logger.info(
            "%s: %d images copied, %d skipped (already on disk)",
            self.name, copied, skipped,
        )

    def load_categories(self) -> dict[str, int] | None:
        self._ensure_categories_loaded()
        return self._categories if self._categories else None

    def num_categories(self) -> int:
        self._ensure_categories_loaded()
        return self._n_categories

    def _ensure_categories_loaded(self) -> None:
        if self._categories_loaded:
            return
        self._categories_loaded = True

        if self.categories_csv is None or not self.categories_csv.exists():
            return

        df = pd.read_csv(self.categories_csv)
        required = {"item_id", "category_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{self.categories_csv} is missing required columns: "
                f"{sorted(missing)}"
            )

        # ``item2idx`` maps external string id -> internal int index.
        # The category dict returned by this method is keyed by the
        # *external* string id (matching :class:`CategoryDataset`'s
        # contract — see DVBPR.load_categories for the parallel).
        _, _, item2idx = self._loaded()
        cats: dict[str, int] = {}
        labels_seen: list = []
        seen_set: set = set()
        for _, row in df.iterrows():
            item_id = str(row["item_id"])
            if item_id not in item2idx:
                continue
            label = int(row["category_label"])
            if label not in seen_set:
                labels_seen.append(label)
                seen_set.add(label)
            cats[item_id] = label

        if not cats:
            return

        label_remap = {label: i for i, label in enumerate(sorted(labels_seen))}
        self._categories = {iid: label_remap[lbl] for iid, lbl in cats.items()}
        self._n_categories = len(label_remap)

    def split(
        self,
        df: pd.DataFrame,
        user2idx: dict[str, int],
        item2idx: dict[str, int],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Default leave-one-out split.

        Subclass and override this method to use a different protocol
        (temporal split, k-fold, etc).  The return value must be a
        triple of dataframes with columns ``user_idx, item_idx``.
        """
        rng = np.random.default_rng(self.seed)

        rows: list[dict] = []
        for _, group in df.groupby("user_id", sort=False):
            if len(group) < self.min_user_interactions:
                continue
            indices = rng.permutation(len(group))
            test_row = group.iloc[indices[0]]
            val_row = group.iloc[indices[1]]
            train_rows = group.iloc[indices[2:]]
            rows.append({"role": "test", "user_id": test_row["user_id"], "item_id": test_row["item_id"]})
            rows.append({"role": "val", "user_id": val_row["user_id"], "item_id": val_row["item_id"]})
            for _, tr in train_rows.iterrows():
                rows.append({"role": "train", "user_id": tr["user_id"], "item_id": tr["item_id"]})

        long_df = pd.DataFrame(rows)
        long_df["user_idx"] = long_df["user_id"].map(user2idx).astype(int)
        long_df["item_idx"] = long_df["item_id"].map(item2idx).astype(int)

        train = long_df[long_df["role"] == "train"][["user_idx", "item_idx"]].reset_index(drop=True)
        val = long_df[long_df["role"] == "val"][["user_idx", "item_idx"]].reset_index(drop=True)
        test = long_df[long_df["role"] == "test"][["user_idx", "item_idx"]].reset_index(drop=True)
        return train, val, test

    def _loaded(self) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
        if self._df is None:
            csv_path = self.raw_dir / "interactions.csv"
            if not csv_path.exists():
                self.download()
            df = pd.read_csv(csv_path)
            required = {"user_id", "item_id", "image_path"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"interactions.csv is missing required columns: "
                    f"{sorted(missing)}"
                )
            df = df.dropna(subset=["user_id", "item_id"])

            users = _stable_unique(df["user_id"])
            items = _stable_unique(df["item_id"])
            self._user2idx = {u: i for i, u in enumerate(users)}
            self._item2idx = {it: i for i, it in enumerate(items)}
            self._df = df
        return self._df, self._user2idx, self._item2idx  # type: ignore[return-value]


def _stable_unique(series: Iterable) -> list[str]:
    """Order-preserving unique that yields a deterministic listing."""
    seen: set = set()
    out: list = []
    for value in series:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
