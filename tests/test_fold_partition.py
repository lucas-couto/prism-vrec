"""Tests for user-level K-fold partitioning (``src.folds.partition``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.folds import (
    EXCLUSION_NO_TARGET,
    EXCLUSION_PROFILE_TOO_SMALL,
    FoldPlan,
    build_fold_plan,
    fold_split,
    load_split_frames,
)

N_USERS = 20
K = 3
MIN_PROFILE = 2


def _splits() -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
    """Twenty users: 0-16 eligible, 17 has two test items, 18 has a tiny profile, 19 no test."""
    train = {u: {u * 10 + j for j in range(3)} for u in range(N_USERS)}
    val = {u: {u * 10 + 3} for u in range(N_USERS)}
    test = {u: {u * 10 + 4} for u in range(N_USERS)}
    test[17] = {174, 175}
    train[18] = set()
    val[18] = {183}
    del test[19]
    return train, val, test


def _plan(seed: int = 7) -> FoldPlan:
    train, val, test = _splits()
    return build_fold_plan(
        train, val, test, n_users=N_USERS, k=K, seed=seed, min_profile=MIN_PROFILE
    )


def test_folds_are_mutually_exclusive_and_cover_every_eligible_user_once() -> None:
    plan = _plan()

    folds = [plan.fold_users(i) for i in range(K)]
    union = [u for fold in folds for u in fold]

    assert len(union) == len(set(union))
    assert set(union) == set(range(17))
    assert set(union) == set(plan.assignment)


def test_fold_sizes_differ_by_at_most_one() -> None:
    plan = _plan()

    sizes = plan.summary()["fold_sizes"]

    assert sum(sizes) == 17
    assert max(sizes) - min(sizes) <= 1


def test_user_with_two_test_items_is_excluded_as_no_target() -> None:
    plan = _plan()

    assert 17 in plan.excluded[EXCLUSION_NO_TARGET]
    assert 17 not in plan.assignment


def test_user_without_test_item_is_excluded_as_no_target() -> None:
    plan = _plan()

    assert 19 in plan.excluded[EXCLUSION_NO_TARGET]


def test_user_with_small_profile_is_excluded_and_summary_reports_counts() -> None:
    plan = _plan()

    summary = plan.summary()

    assert plan.excluded[EXCLUSION_PROFILE_TOO_SMALL] == [18]
    assert summary["excluded"] == {EXCLUSION_PROFILE_TOO_SMALL: 1, EXCLUSION_NO_TARGET: 2}
    assert summary["n_eligible"] == 17
    assert summary["n_users_total"] == N_USERS
    assert summary["k"] == K


def test_fold_split_keeps_held_out_users_out_of_training_and_selection() -> None:
    train, val, test = _splits()
    plan = _plan()

    split = fold_split(plan, 1, train, val, test)

    assert split.test_users == plan.fold_users(1)
    assert not set(split.test_users) & set(split.train_interactions)
    assert not set(split.test_users) & set(split.selection_interactions)


def test_fold_split_target_is_exactly_the_test_item_and_profile_is_train_union_val() -> None:
    train, val, test = _splits()
    plan = _plan()

    split = fold_split(plan, 0, train, val, test)

    for user in split.test_users:
        assert split.target_interactions[user] == test[user]
        assert len(split.target_interactions[user]) == 1
        assert split.profile_interactions[user] == train[user] | val[user]


def test_fold_split_training_users_get_train_only_and_val_as_selection() -> None:
    train, val, test = _splits()
    plan = _plan()

    split = fold_split(plan, 2, train, val, test)

    training_users = set(range(N_USERS)) - set(split.test_users) - {18}
    assert set(split.train_interactions) == training_users
    for user in training_users:
        assert split.train_interactions[user] == train[user]
        assert split.selection_interactions[user] == val[user]
    assert 18 not in split.train_interactions
    assert split.selection_interactions[18] == val[18]


@pytest.mark.parametrize("fold_index", range(K))
def test_fold_split_never_puts_any_test_item_into_training_or_selection(fold_index: int) -> None:
    train, val, test = _splits()
    plan = _plan()

    split = fold_split(plan, fold_index, train, val, test)

    test_items = {item for items in test.values() for item in items}
    train_items = {item for items in split.train_interactions.values() for item in items}
    selection_items = {item for items in split.selection_interactions.values() for item in items}
    assert train_items.isdisjoint(test_items)
    assert selection_items.isdisjoint(test_items)


def test_fold_split_raises_when_target_item_leaks_into_profile() -> None:
    train, val, test = _splits()
    plan = _plan()
    leaking_user = plan.fold_users(0)[0]
    train[leaking_user] = train[leaking_user] | test[leaking_user]

    with pytest.raises(ValueError, match="target"):
        fold_split(plan, 0, train, val, test)


def test_same_seed_gives_same_assignment_and_different_seed_differs() -> None:
    first = _plan(seed=7)
    same = _plan(seed=7)
    other = _plan(seed=8)

    assert first.assignment == same.assignment
    assert first.assignment != other.assignment


def test_invalid_k_raises() -> None:
    train, val, test = _splits()

    with pytest.raises(ValueError, match="k must be"):
        build_fold_plan(train, val, test, n_users=N_USERS, k=1, seed=0, min_profile=1)
    with pytest.raises(ValueError, match="exceeds"):
        build_fold_plan(train, val, test, n_users=N_USERS, k=19, seed=0, min_profile=1)


def test_fold_users_rejects_out_of_range_index() -> None:
    plan = _plan()

    with pytest.raises(ValueError):
        plan.fold_users(K)


def _write_csv(path: Path, rows: list[tuple[int, int]]) -> None:
    lines = ["user_idx,item_idx"] + [f"{u},{i}" for u, i in rows]
    path.write_text("\n".join(lines) + "\n")


def test_load_split_frames_reads_synthetic_csvs(tmp_path: Path) -> None:
    base = tmp_path / "toy"
    base.mkdir()
    _write_csv(base / "train.csv", [(0, 0), (0, 1), (1, 2)])
    _write_csv(base / "val.csv", [(0, 3), (1, 4)])
    _write_csv(base / "test.csv", [(0, 5), (1, 6)])
    (base / "user2idx.json").write_text(json.dumps({"a": 0, "b": 1, "c": 2}))
    (base / "item2idx.json").write_text(json.dumps({str(i): i for i in range(8)}))

    train, val, test, n_users, n_items = load_split_frames(tmp_path, "toy")

    assert train == {0: {0, 1}, 1: {2}}
    assert val == {0: {3}, 1: {4}}
    assert test == {0: {5}, 1: {6}}
    assert (n_users, n_items) == (3, 8)


def test_load_split_frames_falls_back_to_max_index_without_mappings(tmp_path: Path) -> None:
    base = tmp_path / "toy"
    base.mkdir()
    _write_csv(base / "train.csv", [(0, 0), (4, 1)])
    _write_csv(base / "val.csv", [(0, 9)])
    _write_csv(base / "test.csv", [(0, 2)])

    _, _, _, n_users, n_items = load_split_frames(tmp_path, "toy")

    assert (n_users, n_items) == (5, 10)
