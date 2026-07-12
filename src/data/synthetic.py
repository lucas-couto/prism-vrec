"""Synthetic dataset provider for end-to-end smoke tests.

Generates a tiny, deterministic dataset entirely in-process — no
download, no external dependency.  Used to validate that the pipeline
runs from preprocess → statistical without crashing, on any host
(laptop, CI, Mac without GPU).  The numerical results are meaningless
by design; this dataset is for plumbing checks only.

Shape (defaults, all overridable via constructor kwargs):

* 100 users, 200 items, 5 categories
* 8 interactions per user (1 held out as test, 1 as val, rest train)
* 64×64 RGB JPEG images, one per item, deterministic pixel content

Register it under whichever name your config references via
:func:`register_synthetic_provider` (default name ``synthetic``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.base import (
    DatasetProvider,
    register_dataset_provider,
    write_processed_splits,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SyntheticDatasetProvider(DatasetProvider):
    """In-process synthetic dataset for plumbing tests.

    Items, users, interactions and pixel data are all generated from a
    seed so two runs on the same machine produce bit-identical files.
    """

    name = "synthetic"

    def __init__(
        self,
        name: str = "synthetic",
        raw_dir: str | Path | None = None,
        *,
        n_users: int = 100,
        n_items: int = 200,
        n_categories: int = 5,
        interactions_per_user: int = 8,
        image_size: int = 64,
        seed: int = 42,
    ) -> None:
        super().__init__(name=name, raw_dir=raw_dir)
        if interactions_per_user < 3:
            raise ValueError(
                "interactions_per_user must be >= 3 (need train + val + test)"
            )
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.interactions_per_user = interactions_per_user
        self.image_size = image_size
        self.seed = seed

    def download(self) -> None:
        """No-op: the data is generated on demand by the other hooks."""
        logger.info("synthetic dataset has no external source; nothing to download.")

    def save_processed(self, processed_dir: str | Path) -> None:
        """Materialise train/val/test + index mappings + categories sidecar."""
        rng = np.random.default_rng(self.seed)
        # Stringified integer ids so the image-file stems
        # (``<item_idx>.jpg``) match what ``str(item_id)`` produces in
        # the extract step's ``ImageDataset`` lookup.  Same convention
        # the DVBPR provider uses.
        user2idx = {str(i): i for i in range(self.n_users)}
        item2idx = {str(i): i for i in range(self.n_items)}

        train_rows: list[tuple[int, int]] = []
        val_rows: list[tuple[int, int]] = []
        test_rows: list[tuple[int, int]] = []

        for uid in range(self.n_users):
            picks = rng.choice(
                self.n_items,
                size=self.interactions_per_user,
                replace=False,
            ).tolist()
            test_rows.append((uid, int(picks[0])))
            val_rows.append((uid, int(picks[1])))
            for iid in picks[2:]:
                train_rows.append((uid, int(iid)))

        write_processed_splits(
            Path(processed_dir),
            pd.DataFrame(train_rows, columns=["user_idx", "item_idx"]),
            pd.DataFrame(val_rows, columns=["user_idx", "item_idx"]),
            pd.DataFrame(test_rows, columns=["user_idx", "item_idx"]),
            user2idx,
            item2idx,
        )

        cats = rng.integers(0, self.n_categories, size=self.n_items).tolist()
        sidecar = self.raw_dir / "categories.csv"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("w", encoding="utf-8") as fh:
            fh.write("item_id,category_label\n")
            for iid, label in enumerate(cats):
                fh.write(f"{iid},{int(label)}\n")
        logger.info(
            "Wrote %d category rows to %s (%d classes)",
            self.n_items, sidecar, self.n_categories,
        )

    def extract_images(self, image_dir: str | Path) -> None:
        """Generate deterministic ``<item_idx>.jpg`` images."""
        rng = np.random.default_rng(self.seed + 1)
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

        for item_id in range(self.n_items):
            dest = image_dir / f"{item_id}.jpg"
            if dest.exists():
                continue
            pixels = rng.integers(
                0, 256, size=(self.image_size, self.image_size, 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(dest, format="JPEG", quality=80)

        logger.info(
            "Generated %d synthetic JPEG images (%d×%d) under %s",
            self.n_items, self.image_size, self.image_size, image_dir,
        )

    def load_categories(self) -> dict[str, int] | None:
        """Read the sidecar CSV written by :meth:`save_processed`."""
        sidecar = self.raw_dir / "categories.csv"
        if not sidecar.exists():
            return None
        df = pd.read_csv(sidecar)
        unique = sorted(df["category_label"].unique())
        remap = {lbl: i for i, lbl in enumerate(unique)}
        return {
            str(item_id): remap[label]
            for item_id, label in zip(df["item_id"], df["category_label"], strict=True)
        }

    def num_categories(self) -> int:
        """Number of distinct category labels actually present on disk."""
        sidecar = self.raw_dir / "categories.csv"
        if not sidecar.exists():
            return 0
        df = pd.read_csv(sidecar)
        return int(df["category_label"].nunique())


def register_synthetic_provider(name: str = "synthetic") -> None:
    """Register the synthetic provider under ``name`` in the global registry."""
    register_dataset_provider(name, lambda: SyntheticDatasetProvider(name=name))


register_synthetic_provider()
