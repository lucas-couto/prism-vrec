"""Direct unit tests for src.evaluation.metrics.

Every value below is computed by hand so a subtle regression in the
ranking math (off-by-one in the DCG discount, wrong AP denominator,
etc.) fails against a known-good number rather than another code path.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    f1_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RANKED = ["a", "b", "c", "d", "e"]


class TestPrecisionAtK:
    def test_all_hits(self) -> None:
        assert precision_at_k(RANKED, {"a", "b", "c"}, 3) == pytest.approx(1.0)

    def test_partial_hits(self) -> None:
        # hits at positions 1 and 3 -> 2 hits / k=4
        assert precision_at_k(RANKED, {"a", "c"}, 4) == pytest.approx(2 / 4)

    def test_no_hits(self) -> None:
        assert precision_at_k(RANKED, {"z"}, 5) == 0.0

    def test_empty_ranked_list(self) -> None:
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_divides_by_k_not_list_length(self) -> None:
        # 2-item list, k=5: hit count 1 over k=5, not over len=2.
        assert precision_at_k(["a", "z"], {"a"}, 5) == pytest.approx(1 / 5)


class TestRecallAtK:
    def test_leave_one_out_hit(self) -> None:
        assert recall_at_k(RANKED, {"c"}, 3) == 1.0

    def test_leave_one_out_miss(self) -> None:
        assert recall_at_k(RANKED, {"d"}, 3) == 0.0

    def test_partial_recall(self) -> None:
        # 2 of the 4 relevant items are in the top-3
        assert recall_at_k(RANKED, {"a", "b", "x", "y"}, 3) == pytest.approx(2 / 4)

    def test_empty_ground_truth(self) -> None:
        assert recall_at_k(RANKED, set(), 3) == 0.0


class TestF1AtK:
    def test_harmonic_mean(self) -> None:
        # p = 2/4, r = 2/4 -> f1 = 0.5
        assert f1_at_k(RANKED, {"a", "c", "x", "y"}, 4) == pytest.approx(0.5)

    def test_zero_when_no_hits(self) -> None:
        assert f1_at_k(RANKED, {"z"}, 5) == 0.0

    def test_asymmetric_precision_recall(self) -> None:
        # p = 1/2, r = 1/1 -> f1 = 2*(1/2)*1 / (3/2) = 2/3
        assert f1_at_k(["a", "b"], {"a"}, 2) == pytest.approx(2 / 3)


class TestMapAtK:
    def test_reduces_to_reciprocal_rank_leave_one_out(self) -> None:
        # single relevant item at rank 3 -> AP = 1/3
        assert map_at_k(RANKED, {"c"}, 5) == pytest.approx(1 / 3)

    def test_multiple_relevant_items(self) -> None:
        # relevant at ranks 1 and 3: AP = (1/1 + 2/3) / min(5, 2) = 5/6
        assert map_at_k(RANKED, {"a", "c"}, 5) == pytest.approx(5 / 6)

    def test_denominator_is_min_k_gt(self) -> None:
        # 3 relevant, k=2, hits at ranks 1 and 2:
        # AP = (1/1 + 2/2) / min(2, 3) = 1.0
        assert map_at_k(RANKED, {"a", "b", "e"}, 2) == pytest.approx(1.0)

    def test_miss_is_zero(self) -> None:
        assert map_at_k(RANKED, {"z"}, 5) == 0.0

    def test_empty_ground_truth(self) -> None:
        assert map_at_k(RANKED, set(), 5) == 0.0


class TestNdcgAtK:
    def test_hit_at_rank_1_is_perfect(self) -> None:
        assert ndcg_at_k(RANKED, {"a"}, 5) == pytest.approx(1.0)

    def test_leave_one_out_discount(self) -> None:
        # single hit at rank 3: DCG = 1/log2(4), IDCG = 1/log2(2) = 1
        assert ndcg_at_k(RANKED, {"c"}, 5) == pytest.approx(1 / math.log2(4))

    def test_two_relevant_items(self) -> None:
        # hits at ranks 2 and 4; ideal places them at ranks 1 and 2.
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(RANKED, {"b", "d"}, 5) == pytest.approx(dcg / idcg)

    def test_idcg_capped_at_k(self) -> None:
        # 5 relevant items but k=2: IDCG uses min(k, |GT|) = 2 positions,
        # so a ranked list whose top-2 are both relevant scores 1.0.
        assert ndcg_at_k(RANKED, set(RANKED), 2) == pytest.approx(1.0)

    def test_miss_is_zero(self) -> None:
        assert ndcg_at_k(RANKED, {"z"}, 5) == 0.0

    def test_empty_ground_truth(self) -> None:
        assert ndcg_at_k(RANKED, set(), 5) == 0.0


class TestComputeAllMetrics:
    def test_keys_and_values_match_individual_functions(self) -> None:
        gt = {"a", "c"}
        result = compute_all_metrics(RANKED, gt, [1, 3])

        assert set(result) == {
            "precision@1",
            "recall@1",
            "f1@1",
            "map@1",
            "ndcg@1",
            "precision@3",
            "recall@3",
            "f1@3",
            "map@3",
            "ndcg@3",
        }
        assert result["precision@3"] == pytest.approx(precision_at_k(RANKED, gt, 3))
        assert result["ndcg@3"] == pytest.approx(ndcg_at_k(RANKED, gt, 3))

    def test_all_values_in_unit_interval(self) -> None:
        result = compute_all_metrics(RANKED, {"b", "e"}, [1, 5, 20])

        assert all(0.0 <= v <= 1.0 for v in result.values())
