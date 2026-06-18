"""Unit tests for the ACF recommender."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recommenders.acf import ACF

N_USERS = 6
N_ITEMS = 10
N_COMPONENTS = 7
RAW_DIM = 8


def _components() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_ITEMS, N_COMPONENTS, RAW_DIM)).astype("float32")


def _history() -> dict[int, set[int]]:
    return {0: {1, 2, 3}, 1: {4, 5}, 2: set(), 3: set(range(N_ITEMS))}


def _config() -> dict:
    return {"latent_dim": 4, "visual_dim": 5, "att_hidden": 6, "max_history": 4, "l2_reg": 1e-4}


def _model() -> ACF:
    return ACF(
        N_USERS,
        N_ITEMS,
        visual_embeddings=_components(),
        config=_config(),
        train_interactions=_history(),
    )


def test_forward_returns_two_score_vectors() -> None:
    model = _model()

    users = torch.tensor([0, 1, 2, 3])
    pos = torch.tensor([1, 4, 7, 9])
    neg = torch.tensor([8, 9, 0, 1])
    score_pos, score_neg = model(users, pos, neg)

    assert score_pos.shape == (4,)
    assert score_neg.shape == (4,)


def test_bpr_loss_is_finite_and_backpropagates() -> None:
    model = _model()
    users = torch.tensor([0, 1, 2, 3])
    pos = torch.tensor([1, 4, 7, 9])
    neg = torch.tensor([8, 9, 0, 1])

    loss = model.bpr_loss(*model(users, pos, neg))
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(model.user_embedding.weight.grad).all()
    assert torch.isfinite(model.comp_projection.weight.grad).all()


def test_predict_batch_matches_per_user_predict() -> None:
    model = _model().eval()
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        single = model.predict(0, items)
        batched = model.predict_batch(torch.tensor([0, 1]), items)

    assert batched.shape == (2, N_ITEMS)
    assert torch.allclose(batched[0], single, atol=1e-6)


def test_empty_history_user_scores_are_finite() -> None:
    model = _model().eval()
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        scores = model.predict(2, items)  # user 2 has empty history

    assert torch.isfinite(scores).all()


def test_history_buffer_is_padded_and_masked_deterministically() -> None:
    model = _model()

    # user 3 has 10 interactions but max_history=4 -> truncated to sorted first 4
    assert model.history_mask[3].sum().item() == 4
    assert model.history_items[3].tolist() == [0, 1, 2, 3]
    # user 2 has no history -> all masked out
    assert model.history_mask[2].sum().item() == 0


def test_raises_without_train_interactions() -> None:
    with pytest.raises(RuntimeError, match="train_interactions"):
        ACF(N_USERS, N_ITEMS, visual_embeddings=_components(), config=_config())


def test_raises_on_pooled_2d_embeddings() -> None:
    pooled = np.zeros((N_ITEMS, RAW_DIM), dtype="float32")

    with pytest.raises(RuntimeError, match="3-D component"):
        ACF(
            N_USERS,
            N_ITEMS,
            visual_embeddings=pooled,
            config=_config(),
            train_interactions=_history(),
        )
