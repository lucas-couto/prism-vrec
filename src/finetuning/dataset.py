"""Dataset for category classification fine-tuning."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CategoryDataset(Dataset):
    """PyTorch dataset for item category classification.

    Parameters
    ----------
    image_dir:
        Directory containing ``{item_id}.jpg`` images.
    categories:
        Mapping ``{item_id: category_label}``.
    transform:
        Image transform pipeline (from the extractor).
    augment:
        Whether to apply data augmentation (for training split).
    """

    def __init__(
        self,
        image_dir: str | Path,
        categories: dict[str, int],
        transform=None,
        augment: bool = False,
    ) -> None:
        """Categorical-label dataset for fine-tuning.

        Parameters
        ----------
        image_dir:
            Directory containing one image file per item.  Files are
            looked up by *external* item id (file stem) — DVBPR uses
            stringified integers (``"0.jpg"``) and the CSV provider
            uses arbitrary strings (``"i_42.jpg"``); both work because
            this class never tries to parse stems as integers.
        categories:
            Mapping ``{external_item_id (str): category_label (int)}``.
            DVBPR and CSVDatasetProvider both emit this shape via
            :meth:`DatasetProvider.load_categories`.
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        if augment:
            self.aug = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
            )
        else:
            self.aug = None

        # Filter to items that have images on disk.  On distributed
        # filesystems each ``path.exists()`` is a network round-trip,
        # so issuing it once per item costs minutes for a 166K-image
        # dataset.  A single ``os.listdir()`` returns all filenames in
        # one call; set membership is O(1) per item.
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        existing_ids: set[str] = set()
        existing_paths: dict[str, Path] = {}
        try:
            for name in os.listdir(self.image_dir):
                stem, ext = os.path.splitext(name)
                if ext.lower() in valid_exts and stem not in existing_paths:
                    existing_ids.add(stem)
                    existing_paths[stem] = self.image_dir / name
        except (FileNotFoundError, NotADirectoryError):
            pass
        self._paths = existing_paths

        self.items: list[tuple[str, int]] = [
            (str(item_id), label)
            for item_id, label in categories.items()
            if str(item_id) in existing_ids
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        item_id, label = self.items[idx]
        img = Image.open(self._paths[item_id]).convert("RGB")

        if self.aug is not None:
            img = self.aug(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def stratified_split(
        categories: dict[str, int],
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Split categories into train/val with stratification.

        Returns
        -------
        (train_categories, val_categories)
        """
        rng = np.random.default_rng(seed)

        by_label: dict[int, list[str]] = {}
        for item_id, label in categories.items():
            by_label.setdefault(label, []).append(str(item_id))

        train_cats: dict[str, int] = {}
        val_cats: dict[str, int] = {}

        for label, item_ids in by_label.items():
            rng.shuffle(item_ids)
            n_train = max(1, int(len(item_ids) * train_ratio))
            for iid in item_ids[:n_train]:
                train_cats[iid] = label
            for iid in item_ids[n_train:]:
                val_cats[iid] = label

        return train_cats, val_cats
