"""Tests for the sampled evaluation protocol.

Covers pool composition (positives + N negatives, no overlap with
seen or test items), determinism by seed, metric range, and that
``full_ranking`` continues to work unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.protocol import Evaluator


class _ConstantModel:
    """Deterministic stand-in that scores items by their integer id.

    Higher ids → higher score, so the ranking is fully predictable
    and the tests do not depend on a trained recommender.
    """

    def eval(self) -> None:
        pass

    def predict(self, user_id: int, item_ids):
        if isinstance(item_ids, torch.Tensor):
            return item_ids.to(torch.float32)
        return torch.tensor([float(i) for i in item_ids])


def _basic_evaluator(protocol: str = "full_ranking", **kwargs) -> Evaluator:
    train = {0: {0, 1, 2}, 1: {3, 4}, 2: {5}}
    test = {0: {10}, 1: {11}, 2: {12}}
    return Evaluator(
        train_interactions=train,
        test_interactions=test,
        n_items=50,
        k_values=[5, 10],
        protocol=protocol,
        **kwargs,
    )


class TestProtocolValidation:
    def test_rejects_unknown_protocol(self) -> None:
        with pytest.raises(ValueError, match="protocol"):
            _basic_evaluator(protocol="invalid")  # type: ignore[arg-type]

    def test_rejects_zero_negatives_when_sampled(self) -> None:
        with pytest.raises(ValueError, match="n_negatives"):
            _basic_evaluator(protocol="sampled", n_negatives=0)


class TestSampledSampling:
    def test_pool_excludes_seen_and_positives(self) -> None:
        evaluator = _basic_evaluator(protocol="sampled", n_negatives=10)
        for user_id in evaluator.test_users:
            forbidden = (
                evaluator.train_interactions.get(user_id, set())
                | evaluator.test_interactions[user_id]
            )
            negatives = evaluator._sample_negatives(user_id, forbidden)
            assert len(negatives) == 10
            assert not (set(negatives) & forbidden)

    def test_sampling_is_deterministic_by_seed(self) -> None:
        a = _basic_evaluator(protocol="sampled", n_negatives=20, negative_sampling_seed=123)
        b = _basic_evaluator(protocol="sampled", n_negatives=20, negative_sampling_seed=123)
        for user_id in a.test_users:
            forbidden = a.train_interactions.get(user_id, set()) | a.test_interactions[user_id]
            assert a._sample_negatives(user_id, forbidden) == b._sample_negatives(
                user_id, forbidden
            )

    def test_different_seeds_produce_different_pools(self) -> None:
        a = _basic_evaluator(protocol="sampled", n_negatives=20, negative_sampling_seed=1)
        b = _basic_evaluator(protocol="sampled", n_negatives=20, negative_sampling_seed=2)
        forbidden = a.train_interactions[0] | a.test_interactions[0]
        assert a._sample_negatives(0, forbidden) != b._sample_negatives(0, forbidden)

    def test_falls_back_to_all_available_when_pool_too_small(self) -> None:
        # 6 items, forbidden = 4 → only 2 negatives available, asks for 100
        evaluator = Evaluator(
            train_interactions={0: {0, 1, 2}},
            test_interactions={0: {3}},
            n_items=6,
            k_values=[5],
            protocol="sampled",
            n_negatives=100,
        )
        negatives = evaluator._sample_negatives(0, {0, 1, 2, 3})
        assert sorted(negatives) == [4, 5]


class TestSampledEvaluate:
    def test_returns_per_user_dataframe(self) -> None:
        evaluator = _basic_evaluator(protocol="sampled", n_negatives=10)
        df = evaluator.evaluate_per_user(_ConstantModel(), device="cpu")

        assert isinstance(df, pd.DataFrame)
        assert "user_id" in df.columns
        assert len(df) == len(evaluator.test_users)
        metric_cols = [c for c in df.columns if c != "user_id"]
        # Every metric is a finite value in [0, 1]
        values = df[metric_cols].to_numpy(dtype=float)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    def test_does_not_crash_when_user_has_no_train_history(self) -> None:
        evaluator = Evaluator(
            train_interactions={},  # no train history at all
            test_interactions={0: {10}, 1: {11}},
            n_items=50,
            k_values=[5],
            protocol="sampled",
            n_negatives=20,
        )
        df = evaluator.evaluate_per_user(_ConstantModel(), device="cpu")
        assert len(df) == 2


class TestFullRankingStillWorks:
    """Regression: existing full-ranking path is unchanged."""

    def test_default_protocol_is_full_ranking(self) -> None:
        evaluator = _basic_evaluator()
        assert evaluator.protocol == "full_ranking"

    def test_full_ranking_evaluates_against_all_items(self) -> None:
        evaluator = _basic_evaluator(protocol="full_ranking")
        df = evaluator.evaluate_per_user(_ConstantModel(), device="cpu")
        assert len(df) == 3
