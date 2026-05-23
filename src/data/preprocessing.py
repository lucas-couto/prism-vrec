"""Data preprocessing utilities.

Provides k-core filtering, leave-one-out splitting, index-mapping
construction, and a full preprocessing pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def kcore_filter(df: pd.DataFrame, n_min: int = 5) -> pd.DataFrame:
    """Iteratively remove users and items with fewer than *n_min* interactions.

    The filtering is applied in alternating passes (users, then items)
    until no more rows are removed.

    Parameters
    ----------
    df:
        Must contain columns ``user_id`` and ``item_id``.
    n_min:
        Minimum number of interactions required to keep a user or item.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    df = df.copy()
    prev_len = -1

    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= n_min].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= n_min].index
        df = df[df["item_id"].isin(valid_items)]

        logger.debug(
            "k-core iteration %d: %d interactions remaining",
            iteration,
            len(df),
        )

    logger.info(
        "k-core filtering (k=%d) converged after %d iterations: "
        "%d interactions, %d users, %d items",
        n_min,
        iteration,
        len(df),
        df["user_id"].nunique(),
        df["item_id"].nunique(),
    )
    return df.reset_index(drop=True)


def leave_one_out_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using a leave-one-out protocol.

    For each user the *last* interaction goes into the test set and
    everything else goes into the training set.

    - If the DataFrame contains a ``timestamp`` column, interactions are
      sorted by timestamp and the chronologically last one is selected.
    - Otherwise a random interaction is chosen as the test item.

    Parameters
    ----------
    df:
        Must contain at least ``user_id`` and ``item_id``.

    Returns
    -------
    (train_df, test_df)
    """
    has_timestamp = "timestamp" in df.columns

    if has_timestamp:
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        test_idx = df.groupby("user_id").tail(1).index
    else:
        rng = np.random.default_rng(seed=42)
        test_idx_list: list[int] = []
        for _uid, group in df.groupby("user_id"):
            idx = rng.choice(group.index)
            test_idx_list.append(idx)
        test_idx = pd.Index(test_idx_list)

    test_df = df.loc[test_idx].reset_index(drop=True)
    train_df = df.drop(index=test_idx).reset_index(drop=True)

    logger.info(
        "Leave-one-out split: %d train, %d test interactions",
        len(train_df),
        len(test_df),
    )
    return train_df, test_df


def build_mappings(df: pd.DataFrame) -> tuple[dict, dict]:
    """Build contiguous integer mappings for users and items.

    Parameters
    ----------
    df:
        Must contain ``user_id`` and ``item_id`` columns.

    Returns
    -------
    (user2idx, item2idx)
        Dictionaries mapping original IDs to zero-based indices.
    """
    unique_users = sorted(df["user_id"].unique(), key=str)
    unique_items = sorted(df["item_id"].unique(), key=str)

    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}

    logger.info(
        "Built mappings: %d users, %d items",
        len(user2idx),
        len(item2idx),
    )
    return user2idx, item2idx


