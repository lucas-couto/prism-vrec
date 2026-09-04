"""User-level K-fold partitioning with a fold-in evaluation protocol.

Users are split into ``k`` mutually exclusive folds.  On fold ``k_i`` the
users assigned to that fold are held out of training entirely; the other
``k - 1`` folds form the training pool.  Each held-out user is later
folded in from a *profile* (``train ∪ val``) and evaluated on a *target*
(the single item already in the ``test`` split, so results stay
comparable with the leave-one-out protocol).  Training users keep
``train`` alone as history and their ``val`` item for early stopping;
their ``test`` item stays out of training for every user, exactly as
the sequential protocol does.

This module never touches the filesystem: it receives the three
per-user split dicts (see :mod:`src.folds.splits_io`) and only uses the
``test`` split to build the held-out users' targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.folds.splits_io import Interactions, load_split_frames
from src.utils.splits import assert_holdout_disjoint

__all__ = [
    "EXCLUSION_NO_TARGET",
    "EXCLUSION_PROFILE_TOO_SMALL",
    "EXCLUSION_REASONS",
    "FoldPlan",
    "FoldSplit",
    "build_fold_plan",
    "fold_split",
    "load_split_frames",  # re-exported for callers importing it from here
]

EXCLUSION_NO_TARGET = "no_target"
EXCLUSION_PROFILE_TOO_SMALL = "profile_too_small"
EXCLUSION_REASONS: tuple[str, ...] = (EXCLUSION_PROFILE_TOO_SMALL, EXCLUSION_NO_TARGET)


@dataclass(frozen=True)
class FoldPlan:
    """Deterministic user → fold assignment plus the exclusion ledger.

    :param k: Number of folds.
    :param seed: Seed used to shuffle the eligible users.
    :param min_profile: Minimum ``|train ∪ val|`` for a user to be evaluable.
    :param assignment: ``{user_idx: fold_index}`` for eligible users only.
    :param excluded: ``{reason: [user_idx, ...]}`` for ineligible users.
    :param n_users_total: Size of the user index space (``n_users``).
    """

    k: int
    seed: int
    min_profile: int
    assignment: dict[int, int]
    excluded: dict[str, list[int]]
    n_users_total: int

    def fold_users(self, k_index: int) -> list[int]:
        """Sorted users held out (evaluated) on fold ``k_index``.

        :param k_index: Fold index in ``[0, k)``.
        :returns: Sorted ``user_idx`` values assigned to that fold.
        :raises ValueError: If ``k_index`` is outside ``[0, k)``.
        """
        if not 0 <= k_index < self.k:
            raise ValueError(f"fold index {k_index} outside [0, {self.k})")
        return sorted(u for u, f in self.assignment.items() if f == k_index)

    def summary(self) -> dict:
        """Manifest-friendly counts per fold and per exclusion reason."""
        return {
            "k": self.k,
            "seed": self.seed,
            "min_profile": self.min_profile,
            "n_users_total": self.n_users_total,
            "n_eligible": len(self.assignment),
            "fold_sizes": [len(self.fold_users(i)) for i in range(self.k)],
            "excluded": {
                reason: len(self.excluded.get(reason, [])) for reason in EXCLUSION_REASONS
            },
        }


@dataclass(frozen=True)
class FoldSplit:
    """Materialised interactions for one fold.

    :param fold_index: Which fold of the plan this split realises.
    :param train_interactions: Training users → ``train`` history (test item excluded).
    :param selection_interactions: Training users → ``val`` item(s) for early stopping.
    :param profile_interactions: Held-out users → ``train ∪ val`` (fold-in profile).
    :param target_interactions: Held-out users → exactly one target item (``test``).
    :param test_users: Sorted held-out users of this fold.
    """

    fold_index: int
    train_interactions: Interactions
    selection_interactions: Interactions
    profile_interactions: Interactions
    target_interactions: Interactions
    test_users: list[int]


def _profile_of(user: int, train: Interactions, val: Interactions) -> set[int]:
    return train.get(user, set()) | val.get(user, set())


def _classify_user(
    user: int, train: Interactions, val: Interactions, test: Interactions, min_profile: int
) -> str | None:
    """Return the exclusion reason for ``user`` or ``None`` when eligible."""
    if len(test.get(user, set())) != 1:
        return EXCLUSION_NO_TARGET
    if len(_profile_of(user, train, val)) < min_profile:
        return EXCLUSION_PROFILE_TOO_SMALL
    return None


def build_fold_plan(
    train: Interactions,
    val: Interactions,
    test: Interactions,
    *,
    n_users: int,
    k: int,
    seed: int,
    min_profile: int,
) -> FoldPlan:
    """Assign every eligible user to one of ``k`` balanced folds.

    A user is eligible when it has exactly one ``test`` item (the target)
    and ``|train ∪ val| >= min_profile``.  Eligible users are shuffled
    with ``numpy.random.default_rng(seed)`` and dealt round-robin, so
    fold sizes differ by at most one.

    :param train: Per-user training items.
    :param val: Per-user validation items.
    :param test: Per-user test items.
    :param n_users: Size of the user index space; users are ``range(n_users)``.
    :param k: Number of folds (``>= 2``).
    :param seed: Shuffle seed.
    :param min_profile: Minimum fold-in profile size.
    :returns: The :class:`FoldPlan`.
    :raises ValueError: If ``k < 2`` or ``k`` exceeds the number of eligible users.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if min_profile < 1:
        raise ValueError(f"min_profile must be >= 1, got {min_profile}")

    excluded: dict[str, list[int]] = {reason: [] for reason in EXCLUSION_REASONS}
    eligible: list[int] = []
    for user in range(n_users):
        reason = _classify_user(user, train, val, test, min_profile)
        if reason is None:
            eligible.append(user)
        else:
            excluded[reason].append(user)

    if k > len(eligible):
        raise ValueError(f"k={k} exceeds the {len(eligible)} eligible user(s)")

    order = np.random.default_rng(seed).permutation(len(eligible))
    assignment = {eligible[int(pos)]: slot % k for slot, pos in enumerate(order)}
    return FoldPlan(
        k=k,
        seed=seed,
        min_profile=min_profile,
        assignment=assignment,
        excluded=excluded,
        n_users_total=n_users,
    )


def fold_split(
    plan: FoldPlan,
    fold_index: int,
    train: Interactions,
    val: Interactions,
    test: Interactions,
    dataset_name: str = "dataset",
) -> FoldSplit:
    """Materialise the training / fold-in interactions for one fold.

    Held-out users (``plan.fold_users(fold_index)``) get a profile of
    ``train ∪ val`` and a target of their single ``test`` item.  Every
    other user — including users excluded from evaluation — stays in
    the training pool with ``train`` alone as history and ``val`` as
    the selection split; no user's ``test`` item ever enters training.

    :param plan: The fold plan.
    :param fold_index: Fold to materialise.
    :param train: Per-user training items.
    :param val: Per-user validation items.
    :param test: Per-user test items.
    :param dataset_name: For error messages only.
    :returns: The :class:`FoldSplit`.
    :raises ValueError: If a target item leaks into its profile or a held-out
        user leaks into the training pool.
    """
    test_users = plan.fold_users(fold_index)
    held_out = set(test_users)

    profile = {u: _profile_of(u, train, val) for u in test_users}
    target = {u: set(test[u]) for u in test_users}
    assert_holdout_disjoint(profile, target, dataset_name, holdout_name="target")

    train_pool = {u for u in range(plan.n_users_total) if u not in held_out}
    train_interactions = {u: set(train[u]) for u in train_pool if train.get(u)}
    selection = {u: set(val[u]) for u in train_pool if val.get(u)}

    leaked = held_out & (set(train_interactions) | set(selection))
    if leaked:
        raise ValueError(
            f"Dataset {dataset_name!r}, fold {fold_index}: held-out user(s) "
            f"{sorted(leaked)[:5]} leaked into the training pool"
        )
    return FoldSplit(
        fold_index=fold_index,
        train_interactions=train_interactions,
        selection_interactions=selection,
        profile_interactions=profile,
        target_interactions=target,
        test_users=test_users,
    )
