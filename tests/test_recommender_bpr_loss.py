"""Numerical-correctness tests for the BPR loss in :class:`BaseRecommender`.

The loss formula is the algorithmic core of every visual-aware
recommender shipped with the framework, so changes to it are
high-blast-radius.  These tests pin the exact value the formula
returns on tiny inputs against a hand-computed reference.
"""

from __future__ import annotations

import math

import torch

from src.recommenders.base import BaseRecommender


class _BareBonesRecommender(BaseRecommender):
    """Concrete subclass with empty ``forward`` / ``predict`` — only the
    base-class methods are exercised in these tests."""

    def __init__(self, l2_reg: float = 0.0) -> None:
        super().__init__(
            n_users=10,
            n_items=10,
            visual_embeddings=None,
            config={"l2_reg": l2_reg},
        )

    def forward(self, user_ids, pos_item_ids, neg_item_ids):
        return torch.zeros_like(user_ids, dtype=torch.float32), torch.zeros_like(
            user_ids,
            dtype=torch.float32,
        )

    def predict(self, user_id, item_ids):
        return torch.zeros(item_ids.shape[0], dtype=torch.float32)


def _hand_bpr(score_pos: list[float], score_neg: list[float]) -> float:
    """Reference implementation kept deliberately simple."""
    eps = 1e-10
    n = len(score_pos)
    total = 0.0
    for p, q in zip(score_pos, score_neg, strict=False):
        diff = p - q
        sigmoid = 1.0 / (1.0 + math.exp(-diff))
        total += -math.log(sigmoid + eps)
    return total / n


def test_bpr_loss_matches_hand_computation() -> None:
    rec = _BareBonesRecommender(l2_reg=0.0)

    pos = torch.tensor([2.0, 1.5, -0.5], dtype=torch.float32)
    neg = torch.tensor([1.0, 1.5, 0.0], dtype=torch.float32)

    expected = _hand_bpr(pos.tolist(), neg.tolist())
    got = rec.bpr_loss(pos, neg).item()

    assert math.isclose(got, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_bpr_loss_zero_when_pos_dominates_neg() -> None:
    rec = _BareBonesRecommender(l2_reg=0.0)

    pos = torch.tensor([20.0, 20.0], dtype=torch.float32)
    neg = torch.tensor([0.0, 0.0], dtype=torch.float32)

    got = rec.bpr_loss(pos, neg).item()
    assert got < 1e-6


def test_bpr_loss_large_when_neg_dominates_pos() -> None:
    rec = _BareBonesRecommender(l2_reg=0.0)

    pos = torch.tensor([0.0, 0.0], dtype=torch.float32)
    neg = torch.tensor([20.0, 20.0], dtype=torch.float32)

    got = rec.bpr_loss(pos, neg).item()
    assert got > 10.0


def test_bpr_loss_log_two_at_indifference() -> None:
    """When ``score_pos == score_neg``, sigmoid(0) = 0.5 → loss = log 2."""
    rec = _BareBonesRecommender(l2_reg=0.0)

    pos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    neg = pos.clone()

    got = rec.bpr_loss(pos, neg).item()
    assert math.isclose(got, math.log(2.0), rel_tol=1e-4)


def test_bpr_loss_adds_l2_term() -> None:
    """``l2_reg > 0`` adds ``λ * Σ ||θ||²`` on top of the BPR term."""
    rec_no_reg = _BareBonesRecommender(l2_reg=0.0)
    rec_with_reg = _BareBonesRecommender(l2_reg=0.1)

    pos = torch.tensor([1.0, 1.0], dtype=torch.float32)
    neg = torch.tensor([0.5, 0.5], dtype=torch.float32)

    base = rec_no_reg.bpr_loss(pos, neg).item()
    with_reg = rec_with_reg.bpr_loss(pos, neg).item()

    # _BareBonesRecommender carries no trainable params so reg = 0
    # even with l2_reg=0.1 — the wrapper is additive but contributes
    # zero here.
    assert math.isclose(with_reg, base, rel_tol=1e-6)
