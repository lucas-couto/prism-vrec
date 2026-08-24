"""R3: non-finite model scores must fail loud in every ranking path.

Without the guard the paths disagree silently: the batched stable sort
ranks NaN at the TOP (a NaN held-out becomes a spurious ``_rank`` of 1,
inflating metrics) while numpy's lexsort pushes NaN to the bottom.
Masking is the Evaluator's own job, so any NaN/inf in a model's raw
output is a numerical bug (typically fp16/AMP overflow).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.protocol import Evaluator, NonFiniteScoresError


class _IdScoreModel:
    """Finite baseline: scores every item by its integer id."""

    def eval(self) -> None:
        pass

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        return item_ids.to(torch.float32)


class _NaNModel(_IdScoreModel):
    """Emits one NaN score (item 7) for every user."""

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        scores = item_ids.to(torch.float32)
        scores[item_ids == 7] = float("nan")
        return scores


class _NaNBatchModel(_NaNModel):
    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        scores = item_ids.to(torch.float32).repeat(len(user_ids), 1)
        scores[:, 7] = float("nan")
        return scores


class _InfBatchModel(_IdScoreModel):
    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        scores = item_ids.to(torch.float32).repeat(len(user_ids), 1)
        scores[:, 3] = float("inf")
        return scores


def _evaluator(**kwargs) -> Evaluator:
    train = {0: {0, 1}, 1: {2}}
    test = {0: {10}, 1: {11}}
    return Evaluator(
        train_interactions=train,
        test_interactions=test,
        n_items=30,
        k_values=[5],
        **kwargs,
    )


class TestNaNGuard:
    def test_batched_path_raises_on_nan(self) -> None:
        with pytest.raises(NonFiniteScoresError, match="batched path"):
            _evaluator().evaluate_per_user(_NaNBatchModel(), device="cpu")

    def test_batched_path_raises_on_inf(self) -> None:
        # +inf is as pathological as NaN: it would pin the item to the top.
        with pytest.raises(NonFiniteScoresError, match="batched path"):
            _evaluator().evaluate_per_user(_InfBatchModel(), device="cpu")

    def test_single_path_raises_on_nan(self) -> None:
        with pytest.raises(NonFiniteScoresError, match="single path, user 0"):
            _evaluator().evaluate_per_user(_NaNModel(), device="cpu")

    def test_sampled_path_raises_on_nan(self) -> None:
        # Item 7 always lands in the pool: it is a train item of no user,
        # so it is an eligible negative — with n_negatives >= available
        # non-forbidden items the pool is exhaustive.
        with pytest.raises(NonFiniteScoresError, match="sampled path"):
            _evaluator(protocol="sampled", n_negatives=40).evaluate_per_user(
                _NaNModel(), device="cpu"
            )

    def test_finite_scores_still_evaluate(self) -> None:
        df = _evaluator().evaluate_per_user(_IdScoreModel(), device="cpu")

        assert len(df) == 2
        assert np.isfinite(df.drop(columns=["user_id"]).to_numpy()).all()

    def test_guard_precedes_mask_not_confused_by_it(self) -> None:
        # The Evaluator's own -inf train mask must NOT trip the guard:
        # finite raw scores + masking → normal evaluation.
        evaluator = _evaluator()

        result = evaluator.evaluate(_IdScoreModel(), device="cpu")

        assert all(np.isfinite(v) for v in result.values())
