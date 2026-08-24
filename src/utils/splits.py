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


def assert_holdout_disjoint(
    seen_interactions: dict[int, set[int]],
    holdout_interactions: dict[int, set[int]],
    dataset_name: str,
    holdout_name: str = "test",
) -> None:
    """Guard: the held-out split must be per-user disjoint from the seen set.

    The evaluator masks every seen item to ``-inf`` before ranking, so a
    held-out item that ALSO appears in the same user's seen set becomes
    unhittable — that user silently scores 0 on every metric.  Used for
    ``test ∩ (train ∪ val)`` at final evaluation (A3) and for
    ``val ∩ train`` when the selection evaluator is built (R6): a
    duplicated pair would silently deflate the validation metric and
    could steer hyperparameter selection.

    :param seen_interactions: Per-user items masked before ranking.
    :param holdout_interactions: Per-user held-out items being evaluated.
    :param dataset_name: For the error message.
    :param holdout_name: Split name for the error message (``"test"``,
        ``"validation"``).
    :raises ValueError: On the first overlap found, listing examples.
    """
    violations = [
        (user, item)
        for user, items in holdout_interactions.items()
        for item in sorted(items & seen_interactions.get(user, set()))
    ]
    if not violations:
        return
    examples = ", ".join(f"(user={u}, item={i})" for u, i in violations[:5])
    raise ValueError(
        f"Dataset {dataset_name!r}: {len(violations)} (user, item) pair(s) "
        f"appear in BOTH the {holdout_name} split and the same user's seen "
        f"history, e.g. {examples}. Seen items are masked to -inf before "
        f"ranking, so these held-outs can never be hit and the affected "
        f"users would silently score 0. Deduplicate the splits upstream."
    )
