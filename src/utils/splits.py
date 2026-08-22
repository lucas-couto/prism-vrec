"""Readers for the canonical processed splits.

The interaction splits written by ``preprocess`` are consumed by several
steps that must agree on *exactly* which items count as "training data".
Anything fit on item features — a PCA alignment inside fusion, a fixed
projection at extraction time — has to be fit on this set and no other,
or items that occur only in validation/test leak into the basis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def train_item_indices(processed_dir: str | Path, dataset_name: str) -> list[int]:
    """Item indices with at least one *training* interaction.

    The canonical fit set for anything learned from item features.
    Fitting on the full catalogue instead would leak items that appear
    only in validation or test interactions.

    :param processed_dir: Root of the processed splits (``paths.data_processed``).
    :param dataset_name: Dataset whose ``train.csv`` is read.
    :returns: Sorted, de-duplicated ``item_idx`` values.
    """
    train_csv = Path(processed_dir) / dataset_name / "train.csv"
    df = pd.read_csv(train_csv, usecols=["item_idx"])
    return sorted(int(i) for i in df["item_idx"].unique())
