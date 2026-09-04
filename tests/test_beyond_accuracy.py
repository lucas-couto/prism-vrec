"""Unit tests for src.evaluation.beyond_accuracy (hand-computed values).

Formulas under test: EFD (Vargas & Castells 2011, eq. 14 reduced to
Mean Self-Information), ILD (eq. 16 with the cosine normalised to
[0, 1] before the complement), item coverage and category entropy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.beyond_accuracy import (
    RANK_DISCOUNT_BASE,
    catalog_coverage_at_k,
    category_entropy_at_k,
    compute_user_beyond_accuracy,
    efd_at_k,
    efd_excluded_frac_at_k,
    ild_at_k,
)

# pop = [1/2, 1/4, 1/8, 0]: self-information 1, 2, 3 bits (item 3 unseen).
_POP = np.array([0.5, 0.25, 0.125, 0.0])


class TestEfdExcludedFrac:
    def test_should_report_the_share_of_zero_popularity_items_in_the_top_k(self) -> None:
        # Item 3 is unseen in train: 1 of the 3 top slots is excluded from EFD.
        assert efd_excluded_frac_at_k([0, 3, 1], _POP, 3) == pytest.approx(1 / 3)

    def test_should_be_zero_when_every_item_has_train_popularity(self) -> None:
        assert efd_excluded_frac_at_k([0, 1, 2], _POP, 3) == 0.0

    def test_should_be_one_when_every_item_is_cold(self) -> None:
        assert efd_excluded_frac_at_k([3, 3], _POP, 2) == 1.0
        assert math.isnan(efd_at_k([3, 3], _POP, 2))

    def test_should_be_nan_for_an_empty_list(self) -> None:
        assert math.isnan(efd_excluded_frac_at_k([], _POP, 5))

    def test_should_be_emitted_next_to_efd_per_user(self) -> None:
        embeddings = np.eye(4, dtype=np.float32)

        out = compute_user_beyond_accuracy([0, 3, 1], _POP, embeddings, None, [3])

        assert out["efd_excluded_frac@3"] == pytest.approx(1 / 3)
        assert out["efd@3"] == pytest.approx(1.5)


class TestEfdAtK:
    def test_mean_self_information(self) -> None:
        # (1 + 2 + 3) / 3 bits.
        assert efd_at_k([0, 1, 2], _POP, 3) == pytest.approx(2.0)

    def test_truncates_to_k(self) -> None:
        # Only items 0 and 1 count: (1 + 2) / 2.
        assert efd_at_k([0, 1, 2], _POP, 2) == pytest.approx(1.5)

    def test_zero_popularity_item_excluded(self) -> None:
        # Item 3 (pop = 0, unseen in train) is excluded; mean over the rest.
        assert efd_at_k([0, 3, 1], _POP, 3) == pytest.approx(1.5)

    def test_all_zero_popularity_is_nan(self) -> None:
        assert math.isnan(efd_at_k([3, 3], _POP, 2))

    def test_empty_list_is_nan(self) -> None:
        assert math.isnan(efd_at_k([], _POP, 5))

    def test_rank_relevance_discount(self) -> None:
        # disc(n) = 0.85^n over 0-indexed ranks, normalised.
        d = RANK_DISCOUNT_BASE
        expected = (1.0 * 1 + d * 2 + d**2 * 3) / (1 + d + d**2)
        assert efd_at_k([0, 1, 2], _POP, 3, use_rank_relevance=True) == pytest.approx(expected)

    def test_rank_relevance_discount_keeps_original_positions(self) -> None:
        # The zero-pop item at rank 0 is excluded but ranks 1 and 2 keep
        # their ORIGINAL discounts (positions are list positions, not
        # positions among the surviving items).
        d = RANK_DISCOUNT_BASE
        expected = (d * 1 + d**2 * 2) / (d + d**2)
        assert efd_at_k([3, 0, 1], _POP, 3, use_rank_relevance=True) == pytest.approx(expected)


class TestIldAtK:
    def test_identical_items_have_zero_distance(self) -> None:
        emb = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert ild_at_k([0, 1, 2], emb, 3) == pytest.approx(0.0)

    def test_orthogonal_pair_is_half(self) -> None:
        # cos = 0 -> normalised sim 0.5 -> distance 0.5.
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert ild_at_k([0, 1], emb, 2) == pytest.approx(0.5)

    def test_opposite_pair_is_one(self) -> None:
        # cos = -1 (negative cosine!) -> normalised sim 0 -> distance 1,
        # never a distance above 1 as the raw complement would give.
        emb = np.array([[1.0, 0.0], [-1.0, 0.0]])
        assert ild_at_k([0, 1], emb, 2) == pytest.approx(1.0)

    def test_hand_computed_three_items(self) -> None:
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # pairs: (0,1) cos 0 -> 0.5; (0,2) cos 1/sqrt2; (1,2) cos 1/sqrt2.
        d_diag = 1.0 - (1.0 / math.sqrt(2) + 1.0) / 2.0
        expected = (0.5 + 2 * d_diag) / 3
        assert ild_at_k([0, 1, 2], emb, 3) == pytest.approx(expected)

    def test_single_item_list_is_nan(self) -> None:
        # ILD is undefined for one item: NaN, never forced to 0.
        emb = np.eye(3)
        assert math.isnan(ild_at_k([0], emb, 5))

    def test_zero_norm_rows_excluded(self) -> None:
        emb = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert ild_at_k([0, 1, 2], emb, 3) == pytest.approx(0.5)
        assert math.isnan(ild_at_k([0, 1], emb, 2))  # only one valid vector


class TestCatalogCoverage:
    def test_union_over_users(self) -> None:
        tops = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        assert catalog_coverage_at_k(tops, 10, 3) == pytest.approx(5 / 10)

    def test_truncates_to_k(self) -> None:
        tops = [[0, 1, 2], [0, 3, 4]]
        assert catalog_coverage_at_k(tops, 10, 1) == pytest.approx(1 / 10)

    def test_empty_catalog_is_nan(self) -> None:
        assert math.isnan(catalog_coverage_at_k([[0]], 0, 5))


class TestCategoryEntropy:
    def test_uniform_two_categories_is_one_bit(self) -> None:
        cats = np.array([0, 0, 1, 1])
        assert category_entropy_at_k([0, 1, 2, 3], cats, 4) == pytest.approx(1.0)

    def test_single_category_is_zero(self) -> None:
        cats = np.array([7, 7, 7])
        assert category_entropy_at_k([0, 1, 2], cats, 3) == pytest.approx(0.0)

    def test_hand_computed_skewed(self) -> None:
        # 3 of category 0, 1 of category 1 at k=4.
        cats = np.array([0, 0, 0, 1])
        p = np.array([0.75, 0.25])
        expected = float(-(p * np.log2(p)).sum())
        assert category_entropy_at_k([0, 1, 2, 3], cats, 4) == pytest.approx(expected)

    def test_no_categories_dataset_is_none(self) -> None:
        # Tradesy contract: no categories -> explicit N/A, never invented.
        assert category_entropy_at_k([0, 1], None, 2) is None

    def test_empty_list_is_nan(self) -> None:
        assert math.isnan(category_entropy_at_k([], np.array([0]), 5))


class TestComputeUserBeyondAccuracy:
    def test_keys_with_categories(self) -> None:
        emb = np.eye(4)
        cats = np.array([0, 0, 1, 1])
        out = compute_user_beyond_accuracy([0, 1, 2], _POP, emb, cats, [2, 3])
        assert set(out) == {
            "efd@2",
            "efd_excluded_frac@2",
            "ild@2",
            "cat_entropy@2",
            "efd@3",
            "efd_excluded_frac@3",
            "ild@3",
            "cat_entropy@3",
        }

    def test_no_cat_entropy_key_without_categories(self) -> None:
        # N/A stays N/A: the key is absent, not NaN and not 0.
        emb = np.eye(4)
        out = compute_user_beyond_accuracy([0, 1, 2], _POP, emb, None, [3])
        assert set(out) == {"efd@3", "efd_excluded_frac@3", "ild@3"}
