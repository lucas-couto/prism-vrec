"""Filesystem side of the K-fold protocol: read the processed splits.

Kept apart from :mod:`src.folds.partition` so the partitioning logic
never names a split file — it receives plain per-user dicts.  The
``test`` split read here is only ever used to build the held-out users'
targets (the evaluation side, next to ``src.steps.evaluate``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

Interactions = dict[int, set[int]]

__all__ = ["Interactions", "load_split_frames"]


def _build_interactions(df: pd.DataFrame) -> Interactions:
    interactions: Interactions = {}
    for u, i in zip(df["user_idx"], df["item_idx"], strict=True):
        interactions.setdefault(int(u), set()).add(int(i))
    return interactions


def _index_size(mapping_path: Path, frames: list[pd.DataFrame], column: str) -> int:
    if mapping_path.exists():
        with open(mapping_path) as f:
            return len(json.load(f))
    return int(max(int(df[column].max()) for df in frames if len(df))) + 1


def load_split_frames(
    processed_dir: str | Path, dataset: str
) -> tuple[Interactions, Interactions, Interactions, int, int]:
    """Read ``train/val/test.csv`` into per-user item sets.

    ``n_users`` / ``n_items`` come from ``user2idx.json`` / ``item2idx.json``
    when present (the canonical layout) and fall back to ``max_idx + 1``.

    :param processed_dir: Root of the processed splits.
    :param dataset: Dataset directory name under ``processed_dir``.
    :returns: ``(train, val, test, n_users, n_items)``.
    """
    base = Path(processed_dir) / dataset
    frames = [pd.read_csv(base / f"{split}.csv") for split in ("train", "val", "test")]
    train, val, test = (_build_interactions(df) for df in frames)
    n_users = _index_size(base / "user2idx.json", frames, "user_idx")
    n_items = _index_size(base / "item2idx.json", frames, "item_idx")
    return train, val, test, n_users, n_items
