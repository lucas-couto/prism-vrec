"""Leave-one-out deduplication of pre-packaged provider splits."""

from __future__ import annotations

import pytest

from src.utils.splits import assert_holdout_disjoint, deduplicate_leave_one_out


def test_should_leave_a_clean_partition_untouched() -> None:
    train = {1: {10, 11}, 2: {20}}
    validation = {1: {12}, 2: {21}}
    test = {1: {13}, 2: {22}}

    clean_train, clean_val, clean_test = deduplicate_leave_one_out(train, validation, test, "clean")

    assert (clean_train, clean_val, clean_test) == (train, validation, test)


def test_should_drop_a_held_out_item_from_the_training_history() -> None:
    train = {1: {10, 11, 12}}
    validation = {1: {11}}
    test = {1: {12}}

    clean_train, clean_val, clean_test = deduplicate_leave_one_out(
        train, validation, test, "duplicated"
    )

    assert clean_train == {1: {10}}
    assert clean_val == {1: {11}}
    assert clean_test == {1: {12}}


def test_should_resolve_a_shared_val_test_item_in_favour_of_test() -> None:
    train = {1: {10, 11}}
    validation = {1: {11}}
    test = {1: {11}}

    clean_train, clean_val, clean_test = deduplicate_leave_one_out(
        train, validation, test, "shared"
    )

    assert clean_train == {1: {10}}
    assert clean_val == {}
    assert clean_test == {1: {11}}


def test_should_drop_users_left_without_any_training_history() -> None:
    train = {1: {10}, 2: {20, 21}}
    validation = {1: {10}, 2: {21}}
    test = {1: {11}, 2: {22}}

    clean_train, clean_val, clean_test = deduplicate_leave_one_out(train, validation, test, "cold")

    assert clean_train == {2: {20}}
    assert 1 not in clean_val
    assert 1 not in clean_test


@pytest.mark.parametrize("holdout_name", ["validation", "test"])
def test_should_satisfy_the_evaluation_guard_after_deduplication(
    holdout_name: str,
) -> None:
    train = {1: {10, 11, 12}, 2: {20, 21}}
    validation = {1: {11}, 2: {21}}
    test = {1: {12}, 2: {21}}

    clean_train, clean_val, clean_test = deduplicate_leave_one_out(
        train, validation, test, "guarded"
    )

    holdout = clean_val if holdout_name == "validation" else clean_test
    seen = clean_train
    if holdout_name == "test":
        seen = {user: items | clean_val.get(user, set()) for user, items in clean_train.items()}

    assert_holdout_disjoint(seen, holdout, "guarded", holdout_name)
