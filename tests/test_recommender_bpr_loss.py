"""Numerical-correctness tests for the BPR loss in :class:`BaseRecommender`.

The loss formula is the algorithmic core of every visual-aware
recommender shipped with the framework, so changes to it are
high-blast-radius.  These tests pin the exact value the formula
returns on tiny inputs against a hand-computed reference.

Regularisation follows BPR-Opt (Rendle et al., 2009): the L2 term
penalises only the parameters touched by each sampled triple —
``Σ_g λ_g (Σ‖gathered rows_g‖² / batch_size + Σ‖shared params_g‖²)`` —
where "gathered rows" are the embedding rows indexed by the batch (per
occurrence, as gathered) and "shared params" are dense parameters every
triple touches (e.g. the visual projection).  Untouched embedding rows
must NOT contribute.  Each parameter group carries its own constant
``λ_g``: BPR-MF's ``λ_W`` (``l2_reg``), ``λ_H+`` (``l2_reg_item_pos``)
and ``λ_H-`` (``l2_reg_item_neg``); VBPR's ``λ_E`` on the projection
(``l2_reg_projection``, default 0 as in the paper).

BPR-MF (Rendle et al., 2009, Section 4.3.1) is ``x̂_ui = ⟨w_u, h_i⟩``
with no item bias — the bias belongs to VBPR (He & McAuley, 2016).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.recommenders.base import BaseRecommender
from src.recommenders.bpr import BPR
from src.recommenders.vbpr import VBPR


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
    """Reference implementation kept deliberately simple (-ln sigmoid, no eps)."""
    n = len(score_pos)
    total = 0.0
    for p, q in zip(score_pos, score_neg, strict=False):
        diff = p - q
        sigmoid = 1.0 / (1.0 + math.exp(-diff))
        total += -math.log(sigmoid)
    return total / n


def _tiny_bpr(l2_reg: float = 0.0, **extra_config: float) -> BPR:
    """BPR-MF with hand-set weights: 2 users, 4 items, k=2.

    ``user_embedding``  = [[1, 0], [0, 2]]
    ``item_embedding``  = [[1, 1], [2, 0], [0, 3], [5, 5]]

    Item 3 (row [5, 5]) is never sampled in the test batches — it exists
    to prove untouched rows do not contribute.  ``extra_config`` lets a
    test set ``l2_reg_item_pos`` / ``l2_reg_item_neg`` explicitly.
    """
    config = {"latent_dim": 2, "l2_reg": l2_reg, **extra_config}
    model = BPR(n_users=2, n_items=4, config=config)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        model.item_embedding.weight.copy_(
            torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [5.0, 5.0]])
        )
    return model


def _tiny_vbpr(config: dict) -> VBPR:
    """VBPR whose embedding tables (and biases, if present) are all zero
    and whose ``visual_projection.weight`` is a 3x2 all-ones matrix.

    Every score is 0 (bpr = log 2), every gathered row has zero norm and
    ‖W_vis‖² = 6, so the loss isolates the projection's λ.
    """
    visual = np.zeros((4, 2), dtype="float32")
    model = VBPR(
        n_users=2,
        n_items=4,
        visual_embeddings=visual,
        config={"latent_dim": 2, "visual_dim": 3, **config},
    )
    with torch.no_grad():
        model.user_embedding.weight.zero_()
        model.item_embedding.weight.zero_()
        model.item_bias.weight.zero_()
        model.visual_user_embedding.weight.zero_()
        # Zero the (planned) visual bias too, whether it lands as an
        # nn.Embedding or a bare nn.Parameter.
        visual_bias = getattr(model, "visual_bias", None)
        if visual_bias is not None:
            torch.nn.init.zeros_(getattr(visual_bias, "weight", visual_bias))
        model.visual_projection.weight.fill_(1.0)
    return model


#: Batch used against ``_tiny_bpr``: triples (u=0, i=0, j=2), (u=1, i=1, j=2).
_USERS = torch.tensor([0, 1])
_POS = torch.tensor([0, 1])
_NEG = torch.tensor([2, 2])  # deliberate duplicate: item 2 negative twice

#: Hand-derived scores of ``_tiny_bpr`` on the batch above (see the
#: golden-value test for the derivation).
_TINY_SCORE_POS = [1.0, 0.0]
_TINY_SCORE_NEG = [0.0, 6.0]

#: Hand-derived gathered squared norms of ``_tiny_bpr`` on the batch.
_TINY_NORM_USER = 5.0  # ‖(1,0)‖² + ‖(0,2)‖²
_TINY_NORM_POS = 6.0  # ‖(1,1)‖² + ‖(2,0)‖²
_TINY_NORM_NEG = 18.0  # ‖(0,3)‖² twice
_TINY_BATCH = 2


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


def test_bpr_loss_without_forward_has_no_gathered_term() -> None:
    """Direct ``bpr_loss`` calls (no recorded batch) apply the shared
    term only — ``_BareBonesRecommender`` carries no parameters at all,
    so the regulariser is exactly zero even with ``l2_reg > 0``."""
    rec_no_reg = _BareBonesRecommender(l2_reg=0.0)
    rec_with_reg = _BareBonesRecommender(l2_reg=0.1)

    pos = torch.tensor([1.0, 1.0], dtype=torch.float32)
    neg = torch.tensor([0.5, 0.5], dtype=torch.float32)

    base = rec_no_reg.bpr_loss(pos, neg).item()
    with_reg = rec_with_reg.bpr_loss(pos, neg).item()

    assert math.isclose(with_reg, base, rel_tol=1e-6)


def test_recorded_batch_is_consumed_by_one_loss_call() -> None:
    """R11: two losses after ONE forward must not reuse a stale batch.

    The first ``bpr_loss`` consumes the batch recorded by the forward
    pre-hook; the second falls back to the shared-only regulariser
    (zero for plain BPR) instead of silently re-penalising the stale
    gathered rows.
    """
    model = _tiny_bpr(l2_reg=0.1)
    pos, neg = model(_USERS, _POS, _NEG)

    first = model.bpr_loss(pos, neg).item()
    second = model.bpr_loss(pos, neg).item()

    plain = _hand_bpr(pos.tolist(), neg.tolist())
    assert first > plain  # gathered term applied once
    assert math.isclose(second, plain, rel_tol=1e-5)  # stale batch NOT reused


def test_bpr_loss_penalises_only_gathered_rows_hand_derived() -> None:
    """Golden value against the tiny hand-set BPR-MF model.

    Scores (see ``_tiny_bpr`` weights; no item bias in BPR-MF):
        s_pos = [γ_0·γ_i0,  γ_1·γ_i1] = [1·1 + 0·1,  0·2 + 2·0] = [1, 0]
        s_neg = [γ_0·γ_i2,  γ_1·γ_i2] = [1·0 + 0·3,  0·0 + 2·3] = [0, 6]
        bpr   = mean(-log σ(1), -log σ(-6))

    Gathered squared norms (per occurrence, as gathered):
        user rows u0, u1:        ‖(1,0)‖² + ‖(0,2)‖²      = 1 + 4  = 5
        pos item rows i0, i1:    ‖(1,1)‖² + ‖(2,0)‖²      = 2 + 4  = 6
        neg item row i2 TWICE:   ‖(0,3)‖² + ‖(0,3)‖²      = 9 + 9  = 18
        total = 29;  / batch_size 2 = 14.5

    With a single ``l2_reg = 0.1`` (``λ_W = λ_H+ = λ_H- = 0.1``) and no
    shared params in plain BPR,
        loss = bpr + 0.1 * 14.5 = bpr + 1.45.

    The duplicate negative (item 2 in both triples) contributes twice —
    the per-triple reading of BPR-Opt.
    """
    model = _tiny_bpr(l2_reg=0.1)

    score_pos, score_neg = model(_USERS, _POS, _NEG)
    got = model.bpr_loss(score_pos, score_neg).item()

    assert score_pos.tolist() == _TINY_SCORE_POS
    assert score_neg.tolist() == _TINY_SCORE_NEG
    expected = _hand_bpr(_TINY_SCORE_POS, _TINY_SCORE_NEG) + 0.1 * 14.5
    assert math.isclose(got, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_bpr_loss_ignores_untouched_embedding_rows() -> None:
    """Rows never gathered by the batch (item 3) must not contribute:
    scaling them by 100x leaves the loss bit-for-bit unchanged."""
    model = _tiny_bpr(l2_reg=0.1)
    baseline = model.bpr_loss(*model(_USERS, _POS, _NEG)).item()

    with torch.no_grad():
        model.item_embedding.weight[3] *= 100.0
    perturbed = model.bpr_loss(*model(_USERS, _POS, _NEG)).item()

    assert perturbed == baseline


def _tiny_bpr_penalty(model: BPR) -> float:
    """Regularisation part of the loss on the fixed batch (loss - plain bpr)."""
    got = model.bpr_loss(*model(_USERS, _POS, _NEG)).item()
    return got - _hand_bpr(_TINY_SCORE_POS, _TINY_SCORE_NEG)


def test_bpr_lambda_pos_penalises_only_positive_item_rows() -> None:
    """``λ_H+`` alone (``λ_W = λ_H- = 0``) penalises just the positive
    rows: 0.1 * 6 / 2."""
    model = _tiny_bpr(l2_reg=0.0, l2_reg_item_pos=0.1, l2_reg_item_neg=0.0)

    penalty = _tiny_bpr_penalty(model)

    expected = 0.1 * _TINY_NORM_POS / _TINY_BATCH
    assert math.isclose(penalty, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_bpr_lambda_neg_penalises_only_negative_item_rows() -> None:
    """``λ_H-`` alone (``λ_W = λ_H+ = 0``) penalises just the negative
    rows (item 2 twice): 0.1 * 18 / 2."""
    model = _tiny_bpr(l2_reg=0.0, l2_reg_item_pos=0.0, l2_reg_item_neg=0.1)

    penalty = _tiny_bpr_penalty(model)

    expected = 0.1 * _TINY_NORM_NEG / _TINY_BATCH
    assert math.isclose(penalty, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_bpr_lambda_user_alone_penalises_only_user_rows() -> None:
    """``λ_W`` alone (both item keys explicitly 0) penalises just the
    user rows: 0.1 * 5 / 2."""
    model = _tiny_bpr(l2_reg=0.1, l2_reg_item_pos=0.0, l2_reg_item_neg=0.0)

    penalty = _tiny_bpr_penalty(model)

    expected = 0.1 * _TINY_NORM_USER / _TINY_BATCH
    assert math.isclose(penalty, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_bpr_missing_item_lambdas_fall_back_to_l2_reg() -> None:
    """Absent ``l2_reg_item_pos`` / ``l2_reg_item_neg`` inherit ``l2_reg``,
    so a single-λ config equals the fully explicit one."""
    implicit = _tiny_bpr(l2_reg=0.1)
    explicit = _tiny_bpr(l2_reg=0.1, l2_reg_item_pos=0.1, l2_reg_item_neg=0.1)

    penalty_implicit = _tiny_bpr_penalty(implicit)
    penalty_explicit = _tiny_bpr_penalty(explicit)

    assert math.isclose(penalty_implicit, penalty_explicit, rel_tol=1e-6)
    assert math.isclose(
        penalty_implicit,
        0.1 * (_TINY_NORM_USER + _TINY_NORM_POS + _TINY_NORM_NEG) / _TINY_BATCH,
        rel_tol=1e-5,
        abs_tol=1e-6,
    )


def test_bpr_mf_has_no_item_bias() -> None:
    """BPR-MF (Rendle et al., 2009) is ``⟨w_u, h_i⟩`` only: no bias
    parameter exists and the item table is the only gathered item group."""
    model = _tiny_bpr()

    assert not hasattr(model, "item_bias")
    assert "item_bias" not in dict(model.named_parameters())
    assert model._L2_ITEM_TABLES == ("item_embedding",)


def test_bpr_predict_batch_matches_predict() -> None:
    """``predict_batch`` is the row-stacked form of ``predict`` and both
    reduce to the plain dot product ``γ_u^T γ_i``."""
    model = _tiny_bpr()
    items = torch.tensor([0, 1, 2, 3])

    batch = model.predict_batch(_USERS, items)
    per_user = torch.stack([model.predict(int(u), items) for u in _USERS])
    dot = model.user_embedding.weight @ model.item_embedding.weight.T

    assert torch.equal(batch, per_user)
    assert torch.equal(batch, dot)


def test_bpr_loss_still_regularises_shared_dense_params() -> None:
    """Dense params every triple touches (VBPR's W_vis) stay penalised
    in full when their λ is set.

    All embedding tables are zeroed, so every score is 0 (bpr = log 2)
    and every gathered row has zero norm.  ``visual_projection.weight``
    is a 3x2 all-ones matrix → ‖W‖² = 6; with an explicit
    ``l2_reg_projection = 0.1`` (``λ_E``),
        loss = log 2 + 0.1 * 6.
    """
    model = _tiny_vbpr({"l2_reg": 0.1, "l2_reg_projection": 0.1})

    got = model.bpr_loss(*model(_USERS, _POS, _NEG)).item()

    expected = math.log(2.0) + 0.1 * 6.0
    assert math.isclose(got, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_vbpr_projection_unpenalised_without_explicit_lambda() -> None:
    """VBPR's paper sets ``λ_E = 0``: with only ``l2_reg`` in the config the
    projection contributes nothing, so the loss is exactly the log-loss
    (log 2 on the all-zero tables)."""
    model = _tiny_vbpr({"l2_reg": 0.1})

    got = model.bpr_loss(*model(_USERS, _POS, _NEG)).item()

    assert math.isclose(got, math.log(2.0), rel_tol=1e-5, abs_tol=1e-6)
