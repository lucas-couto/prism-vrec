"""DVBPR dataset provider.

Loads the pre-packaged .npy datasets from Kang et al. (ICDM 2017):
  - AmazonFashion6ImgPartitioned.npy
  - AmazonWomenWithImgPartitioned.npy
  - AmazonMenWithImgPartitioned.npy
  - TradesyImgPartitioned.npy

Each file contains [user_train, user_validation, user_test, Item, usernum, itemnum].
Images are stored as raw JPEG bytes inside the Item dict.

Source: https://github.com/kang205/DVBPR

This module also registers the four DVBPR datasets with the global
:mod:`src.data.base` registry, so the rest of the pipeline can resolve
them by name without importing this module directly.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.base import DatasetProvider, register_dataset_provider
from src.data.categories import derive_categories, write_categories_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://cseweb.ucsd.edu/~wckang/DVBPR"

DATASETS = {
    "amazon_fashion": "AmazonFashion6ImgPartitioned.npy",
    "amazon_women": "AmazonWomenWithImgPartitioned.npy",
    "amazon_men": "AmazonMenWithImgPartitioned.npy",
    "tradesy": "TradesyImgPartitioned.npy",
}


class DVBPRDataLoader(DatasetProvider):
    """Loader for the DVBPR pre-packaged ``.npy`` datasets."""

    def __init__(self, dataset_name: str, raw_dir: str = "data/raw") -> None:
        if dataset_name not in DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name!r}. "
                f"Available: {list(DATASETS.keys())}"
            )
        super().__init__(name=dataset_name, raw_dir=raw_dir)
        self.dataset_name = dataset_name
        self.filename = DATASETS[dataset_name]
        self.npy_path = self.raw_dir / self.filename

        self._data: list | None = None

    def _expected_remote_size(self, url: str, requests_module) -> int | None:
        """Return the server's reported total size for ``url``, or ``None``."""
        try:
            head = requests_module.head(url, timeout=30, allow_redirects=True)
            head.raise_for_status()
            value = head.headers.get("content-length")
            return int(value) if value else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("HEAD request failed for %s: %s", url, exc)
            return None

    def download(self, max_retries: int = 5) -> None:
        """Download the dataset's ``.npy`` with resume + size validation.

        Three layers of protection against silently truncated files:

        1. The body is streamed into ``<file>.npy.partial`` first.
        2. After streaming, the on-disk size is compared against the
           server's expected size (Content-Length); the rename to the
           final ``.npy`` only happens when they match.
        3. If a previous run left a final ``.npy`` whose size disagrees
           with what the server reports today, it is re-downloaded.
        """
        import requests

        url = f"{_BASE_URL}/{self.filename}"
        partial_path = self.npy_path.with_suffix(".npy.partial")
        expected_size = self._expected_remote_size(url, requests)

        if self.npy_path.exists():
            local_size = self.npy_path.stat().st_size
            if expected_size is None or local_size == expected_size:
                logger.info(
                    "File already exists, skipping download: %s (%.1f MB)",
                    self.npy_path, local_size / 1e6,
                )
                return
            logger.warning(
                "Existing %s is %d bytes but server reports %d — re-downloading.",
                self.npy_path, local_size, expected_size,
            )
            self.npy_path.unlink()

        for attempt in range(1, max_retries + 1):
            downloaded = partial_path.stat().st_size if partial_path.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

            if downloaded:
                logger.info(
                    "Resuming download from %.1f MB (attempt %d/%d)...",
                    downloaded / 1e6, attempt, max_retries,
                )
            else:
                logger.info("Downloading %s from %s ...", self.dataset_name, url)

            try:
                response = requests.get(
                    url, stream=True, timeout=600, headers=headers,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to download {self.dataset_name} from {url}: {exc}"
                    ) from exc
                logger.warning("Connection failed, retrying... (%s)", exc)
                continue

            total_str = response.headers.get("content-length", "0")
            chunk_total = int(total_str) if total_str.isdigit() else 0
            total = downloaded + chunk_total if chunk_total else (expected_size or 0)

            try:
                mode = "ab" if downloaded else "wb"
                with (
                    open(partial_path, mode) as fout,
                    tqdm(
                        total=total or None,
                        initial=downloaded,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {self.dataset_name}",
                    ) as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        fout.write(chunk)
                        pbar.update(len(chunk))

                final_size = partial_path.stat().st_size
                if expected_size is not None and final_size != expected_size:
                    logger.warning(
                        "Truncated download: got %d bytes, expected %d.  "
                        "Keeping %s for the next retry to resume from.",
                        final_size, expected_size, partial_path,
                    )
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"{self.dataset_name}: download truncated after "
                            f"{max_retries} attempts ({final_size} / {expected_size} bytes)"
                        )
                    continue

                partial_path.rename(self.npy_path)
                logger.info(
                    "Saved %s to %s (%.1f MB)",
                    self.dataset_name, self.npy_path, final_size / 1e6,
                )
                return

            except requests.RequestException as exc:
                logger.warning(
                    "Download interrupted at %.1f MB (%s). Will retry...",
                    partial_path.stat().st_size / 1e6, exc,
                )
                if attempt == max_retries:
                    raise

    def _load_raw(self) -> list:
        """Load and cache the raw .npy data."""
        if self._data is not None:
            return self._data

        if not self.npy_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.npy_path}. "
                "Call download() first."
            )

        logger.info("Loading %s (this may take a moment)...", self.npy_path.name)
        self._data = np.load(self.npy_path, allow_pickle=True, encoding="latin1")
        return self._data

    def load_splits(self) -> tuple[dict, dict, dict, int, int]:
        """Load train/validation/test interaction splits.

        Returns
        -------
        train : dict[int, set[int]]
            {user_id: set of item_ids} for training.
        validation : dict[int, set[int]]
            {user_id: {item_id}} — 1 held-out item per user for validation.
        test : dict[int, set[int]]
            {user_id: {item_id}} — 1 held-out item per user for test.
        n_users : int
        n_items : int
        """
        data = self._load_raw()
        user_train_raw, user_val_raw, user_test_raw, _, usernum, itemnum = data

        def _parse_split(raw: dict) -> dict[int, set[int]]:
            result: dict[int, set[int]] = {}
            for uid, interactions in raw.items():
                items = set()
                for entry in interactions:
                    if isinstance(entry, dict):
                        iid = entry.get("productid") or entry.get(b"productid")
                    else:
                        iid = int(entry)
                    if iid is not None:
                        items.add(int(iid))
                if items:
                    result[int(uid)] = items
            return result

        train = _parse_split(user_train_raw)
        validation = _parse_split(user_val_raw)
        test = _parse_split(user_test_raw)

        logger.info(
            "Loaded splits: %d users, %d items, "
            "train=%d interactions, val=%d users, test=%d users",
            int(usernum),
            int(itemnum),
            sum(len(v) for v in train.values()),
            len(validation),
            len(test),
        )
        return train, validation, test, int(usernum), int(itemnum)

    def load_categories(self) -> dict[str, int] | None:
        """Load category labels for each item.

        Resolution order:

        1. **Sidecar CSV** at ``data/raw/<name>/categories.csv`` —
           if present, its contents win over the ``.npy`` ``c`` field.
           This is what enables Battery 2 fine-tuning on DVBPR splits
           that lack the one-hot vector but have the textual taxonomy
           (amazon_men, amazon_women).  :meth:`save_processed` auto-
           derives this sidecar from the McAuley taxonomy embedded in
           the ``.npy`` when the dataset lacks the ``c`` field, so no
           manual pre-processing step is required.
        2. **`Item[i]['c']` vector** baked into the ``.npy`` —
           the canonical DVBPR field used by ``amazon_fashion``.
        3. **None** — Battery 2 falls back to ``tradesy_transfer_from``
           or skips the dataset entirely (configurable in
           ``configs/finetuning.yaml``).

        Item IDs are stringified so the contract matches
        :class:`CategoryDataset` and works uniformly with
        :class:`CSVDatasetProvider`, which emits arbitrary string IDs.
        """
        sidecar = self.raw_dir / "categories.csv"
        if sidecar.exists():
            return self._load_categories_from_csv(sidecar)

        data = self._load_raw()
        _, _, _, items_raw, _, itemnum = data

        categories: dict[str, int] = {}
        for item_id in range(int(itemnum)):
            item = items_raw[item_id]
            cat_vec = None
            if isinstance(item, dict):
                # Avoid `a or b` here: when `a` is a numpy array, Python tries
                # to evaluate its truthiness, which raises ValueError for
                # multi-element arrays.
                cat_vec = item.get("c")
                if cat_vec is None:
                    cat_vec = item.get(b"c")

            if cat_vec is None or (hasattr(cat_vec, "__len__") and len(cat_vec) == 0):
                return None

            categories[str(item_id)] = int(np.argmax(cat_vec))

        n_classes = len(set(categories.values()))
        logger.info(
            "Loaded categories for %d items (%d classes)", len(categories), n_classes,
        )
        return categories

    def num_categories(self) -> int:
        """Return the number of category classes, or 0 if no categories.

        Honours the sidecar CSV when present (mirrors
        :meth:`load_categories`) so derived taxonomies declare their own
        ``n_classes``.
        """
        sidecar = self.raw_dir / "categories.csv"
        if sidecar.exists():
            cats = self._load_categories_from_csv(sidecar)
            if cats:
                return len(set(cats.values()))

        data = self._load_raw()
        _, _, _, items_raw, _, _ = data
        item = items_raw[0]
        cat_vec = None
        if isinstance(item, dict):
            cat_vec = item.get("c")
            if cat_vec is None:
                cat_vec = item.get(b"c")
        if cat_vec is None or (hasattr(cat_vec, "__len__") and len(cat_vec) == 0):
            return 0
        return len(cat_vec)

    def _load_categories_from_csv(self, sidecar) -> dict[str, int] | None:
        """Load a sidecar ``categories.csv`` as ``{str item_id: int label}``.

        Same format the framework expects from
        :class:`src.data.example_csv.CSVDatasetProvider`: two columns
        ``item_id`` and ``category_label``.  Labels are remapped to a
        contiguous ``[0, n_classes)`` range so callers do not need to
        worry about discontinuities.
        """
        import pandas as pd

        df = pd.read_csv(sidecar)
        required = {"item_id", "category_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{sidecar} is missing required columns: {sorted(missing)}",
            )
        if df.empty:
            return None

        unique_labels = sorted(df["category_label"].unique())
        remap = {lbl: i for i, lbl in enumerate(unique_labels)}
        return {
            str(row["item_id"]): remap[row["category_label"]]
            for _, row in df.iterrows()
        }

    def extract_images(self, image_dir: str | Path) -> None:
        """Extract embedded JPEG images to disk.

        Each item's image is saved as ``{image_dir}/{item_id}.jpg``.
        Already-existing images are skipped.

        Parameters
        ----------
        image_dir:
            Directory where images will be saved.
        """
        data = self._load_raw()
        _, _, _, items_raw, _, itemnum = data

        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

        extracted = 0
        skipped = 0
        no_image = 0
        errors = 0

        first_item = items_raw[0]
        logger.info(
            "Item structure: type=%s, keys/attrs=%s",
            type(first_item).__name__,
            list(first_item.keys()) if hasattr(first_item, "keys") else dir(first_item)[:10],
        )

        for item_id in tqdm(range(int(itemnum)), desc="Extracting images"):
            dest = image_dir / f"{item_id}.jpg"
            if dest.exists():
                skipped += 1
                continue

            try:
                item = items_raw[item_id]

                img_bytes = None
                if isinstance(item, dict):
                    img_bytes = item.get("imgs") or item.get(b"imgs")
                elif hasattr(item, "imgs"):
                    img_bytes = item.imgs
                elif isinstance(item, (list, np.ndarray)) and len(item) > 0:
                    # Some formats store [jpeg_bytes, category_vector]
                    img_bytes = item[0] if isinstance(item[0], bytes) else None

                if img_bytes is None or len(img_bytes) == 0:
                    no_image += 1
                    continue

                if isinstance(img_bytes, str):
                    img_bytes = img_bytes.encode("latin1")

                dest.write_bytes(img_bytes)
                extracted += 1
            except Exception as exc:  # noqa: BLE001 — count and keep going
                # A genuine extraction failure (corrupt bytes, disk full,
                # permission) is distinct from an item that simply has no
                # image; conflating them under-reports data loss.  Log the
                # first few and always count them separately.
                if errors < 5:
                    logger.warning("Error extracting item %d: %s", item_id, exc)
                errors += 1

        if errors:
            logger.warning("Image extraction hit %d errors (see warnings above).", errors)
        logger.info(
            "Images: %d extracted, %d skipped (existing), %d without image, %d errors",
            extracted,
            skipped,
            no_image,
            errors,
        )

    def save_processed(self, processed_dir: str | Path) -> None:
        """Save train/val/test splits in the pipeline's standard CSV format.

        Creates:
          - train.csv (user_idx, item_idx)
          - val.csv (user_idx, item_idx)
          - test.csv (user_idx, item_idx)
          - user2idx.json (identity mapping, IDs are already contiguous)
          - item2idx.json (identity mapping, IDs are already contiguous)
        """
        import pandas as pd

        train, validation, test, n_users, n_items = self.load_splits()

        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        def _split_to_df(split: dict[int, set[int]]) -> pd.DataFrame:
            rows = []
            for uid, items in split.items():
                for iid in items:
                    rows.append({"user_idx": uid, "item_idx": iid})
            return pd.DataFrame(rows)

        train_df = _split_to_df(train)
        val_df = _split_to_df(validation)
        test_df = _split_to_df(test)

        train_df.to_csv(processed_dir / "train.csv", index=False)
        val_df.to_csv(processed_dir / "val.csv", index=False)
        test_df.to_csv(processed_dir / "test.csv", index=False)

        user2idx = {str(i): i for i in range(n_users)}
        item2idx = {str(i): i for i in range(n_items)}

        with open(processed_dir / "user2idx.json", "w") as f:
            json.dump(user2idx, f)
        with open(processed_dir / "item2idx.json", "w") as f:
            json.dump(item2idx, f)

        logger.info(
            "Saved processed data to %s: train=%d, val=%d, test=%d, users=%d, items=%d",
            processed_dir,
            len(train_df),
            len(val_df),
            len(test_df),
            n_users,
            n_items,
        )

        self._ensure_categories_sidecar()

    def _ensure_categories_sidecar(self) -> None:
        """Derive ``categories.csv`` from textual taxonomy when ``c`` is absent.

        DVBPR splits without the one-hot ``c`` field (amazon_men,
        amazon_women, tradesy) still carry a McAuley textual taxonomy
        per item.  When the sidecar CSV is missing AND the ``.npy``
        lacks ``c``, we materialise the sidecar so fine-tuning has
        canonical categories without a manual pre-processing step.
        Already-present sidecars are left untouched (the researcher's
        choice of level/min_samples wins).
        """
        sidecar = self.raw_dir / "categories.csv"
        if sidecar.exists():
            return

        data = self._load_raw()
        _, _, _, items_raw, _, _ = data
        sample = items_raw[0] if hasattr(items_raw, "__getitem__") else None
        if isinstance(sample, dict):
            has_c = sample.get("c") is not None or sample.get(b"c") is not None
            if has_c:
                return

        items_iter: list[tuple[Any, Any]]
        if isinstance(items_raw, dict):
            items_iter = list(items_raw.items())
        else:
            items_iter = list(enumerate(items_raw))

        mapping = derive_categories(items_iter)
        if mapping is None:
            logger.info(
                "%s: no usable taxonomy found, skipping categories derivation.",
                self.name,
            )
            return
        write_categories_csv(mapping, sidecar)


for _name in DATASETS:
    register_dataset_provider(_name, partial(DVBPRDataLoader, _name))

