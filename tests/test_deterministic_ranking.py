"""Deterministic tie-breaking + Wilcoxon zero handling (v2).

The unified rule: exact-score ties are broken by the random, seed-fixed
``_tiebreak_key`` permutation, in all three ranking paths (batched torch,
sampled numpy, single-user numpy). These tests assert the mechanism (the
all-tied ranking follows the key, not item id) and cross-path agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.protocol import Evaluator
from src.evaluation.statistical import n_nonzero_pairs, wilcoxon_test

N_ITEMS = 10


class _TiedScoresModel(torch.nn.Module):
    """Model whose scores are all identical — pure tie-break test."""

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(item_ids.shape[0])

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(user_ids.shape[0], item_ids.shape[0])


def _evaluator(**kwargs) -> Evaluator:
    return Evaluator(
        train_interactions={0: {9}},
        test_interactions={0: {1}},
        n_items=N_ITEMS,
        k_values=[5],
        **kwargs,
    )


class TestUnifiedTieBreak:
    def test_all_tied_ranks_by_tiebreak_key_not_id(self) -> None:
        evaluator = _evaluator()
        # All scores tie; item 9 is masked (train). Under the random rule
        # the top-5 are the 5 unmasked items with the smallest tie-break
        # key — NOT ids 0..4. The held-out (item 1) is hit iff it is one.
        key = evaluator._tiebreak_key
        unmasked = [i for i in range(N_ITEMS) if i != 9]
        top5 = sorted(unmasked, key=lambda i: key[i])[:5]
        expected = 1.0 if 1 in top5 else 0.0

        results = evaluator.evaluate(_TiedScoresModel(), device="cpu")
        assert results["recall@5"] == expected

    def test_single_user_path_matches_batched_rule(self) -> None:
        evaluator = _evaluator()
        batched = evaluator.evaluate(_TiedScoresModel(), device="cpu")["recall@5"]

        single = evaluator._rank_and_score(0, np.zeros(N_ITEMS, dtype=np.float64))["recall@5"]

        # Both paths apply the same key, so they must agree exactly.
        assert single == batched

    def test_sampled_path_does_not_favor_positives_on_ties(self) -> None:
        # Positive is item 8: under all-tied scores its rank must follow
        # the tie-break key over the pool, never the positives-first pool
        # order (which would inflate metrics).
        evaluator = Evaluator(
            train_interactions={0: set()},
            test_interactions={0: {8}},
            n_items=N_ITEMS,
            k_values=[5],
            protocol="sampled",
            n_negatives=8,
        )
        pool = [8] + evaluator._sample_negatives(0, {8})
        top5 = sorted(pool, key=lambda i: evaluator._tiebreak_key[i])[:5]
        expected = 1.0 if 8 in top5 else 0.0

        results = evaluator.evaluate(_TiedScoresModel(), device="cpu")
        assert results["recall@5"] == expected

    def test_distinct_scores_are_unaffected(self) -> None:
        evaluator = _evaluator()
        scores = np.arange(N_ITEMS, dtype=np.float64)  # strictly increasing

        metrics = evaluator._rank_and_score(0, scores)

        # highest scores are 9 (masked -> -inf), 8, 7, 6, 5, 4
        assert metrics["recall@5"] == 0.0


class TestWilcoxonPratt:
    def test_zero_heavy_pairs_use_pratt(self) -> None:
        # 0/1 per-user metric with a single discordant pair: under the
        # default "wilcox" the effective n collapses to 1; pratt keeps
        # the zeros in the ranking. The p-value must be computable and
        # non-significant for such weak evidence.
        a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])

        stat, p = wilcoxon_test(a, b)

        assert 0.0 <= p <= 1.0
        assert p > 0.05

    def test_identical_arrays_return_degenerate(self) -> None:
        a = np.ones(6)
        assert wilcoxon_test(a, a.copy()) == (0.0, 1.0)

    def test_n_nonzero_pairs_counts_signal(self) -> None:
        a = np.array([1.0, 0.0, 1.0, 0.0])
        b = np.array([1.0, 1.0, 0.0, 0.0])

        assert n_nonzero_pairs(a, b) == 2


class TestPairwiseReportsNPairs:
    def test_columns_present(self) -> None:
        import pandas as pd

        from src.evaluation.statistical import pairwise_significance

        rng = np.random.default_rng(0)
        rows = []
        for cfg in ["m1", "m2"]:
            for uid in range(30):
                rows.append(
                    {"model_name": cfg, "user_id": uid, "ndcg@10": float(rng.random() > 0.5)}
                )
        df = pd.DataFrame(rows)

        out = pairwise_significance(df, metric="ndcg@10")

        assert {"n_pairs", "n_nonzero_pairs"} <= set(out.columns)
        assert (out["n_nonzero_pairs"] <= out["n_pairs"]).all()


@pytest.mark.parametrize("device_scores", ["cpu"])
def test_batched_and_single_paths_agree_under_ties(device_scores: str) -> None:
    """Cross-path consistency: the same tied input yields the same ranking."""
    evaluator = _evaluator()

    batched = evaluator.evaluate(_TiedScoresModel(), device="cpu")
    single = evaluator._rank_and_score(0, np.zeros(N_ITEMS))

    assert batched["recall@5"] == single["recall@5"]
    assert batched["ndcg@5"] == single["ndcg@5"]
