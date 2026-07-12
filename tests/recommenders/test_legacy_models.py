"""Model tests for the non-ACF recommenders (bpr, vbpr, avbpr, deepstyle, vnpr).

Mirrors the contract exercised in test_acf_model.py: forward returns a
pair of score vectors, the BPR loss is finite and backpropagates, and
predict_batch agrees with per-user predict.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recommenders.avbpr import AVBPR
from src.recommenders.bpr import BPR
from src.recommenders.deepstyle import DeepStyle
from src.recommenders.vbpr import VBPR
from src.recommenders.vnpr import VNPR

N_USERS = 6
N_ITEMS = 10
RAW_DIM = 8

MODELS = {
    "bpr": (BPR, {"latent_dim": 4, "l2_reg": 1e-4}, False),
    "vbpr": (VBPR, {"latent_dim": 4, "visual_dim": 5, "l2_reg": 1e-4}, True),
    "avbpr": (
        AVBPR,
        {"latent_dim": 4, "visual_dim": 5, "att_hidden": 6, "l2_reg": 1e-4},
        True,
    ),
    "deepstyle": (DeepStyle, {"latent_dim": 4, "style_dim": 5, "l2_reg": 1e-4}, True),
    "vnpr": (VNPR, {"latent_dim": 4, "hidden_layers": [6], "l2_reg": 1e-4}, True),
}


def _visual() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_ITEMS, RAW_DIM)).astype("float32")


def _model(name: str):
    cls, config, needs_visual = MODELS[name]
    visual = _visual() if needs_visual else None
    return cls(N_USERS, N_ITEMS, visual_embeddings=visual, config=config)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_forward_returns_two_score_vectors(name: str) -> None:
    model = _model(name)

    users = torch.tensor([0, 1, 2, 3])
    pos = torch.tensor([1, 4, 7, 9])
    neg = torch.tensor([8, 9, 0, 1])
    score_pos, score_neg = model(users, pos, neg)

    assert score_pos.shape == (4,)
    assert score_neg.shape == (4,)
    assert torch.isfinite(score_pos).all()
    assert torch.isfinite(score_neg).all()


@pytest.mark.parametrize("name", sorted(MODELS))
def test_bpr_loss_is_finite_and_backpropagates(name: str) -> None:
    model = _model(name)
    users = torch.tensor([0, 1, 2, 3])
    pos = torch.tensor([1, 4, 7, 9])
    neg = torch.tensor([8, 9, 0, 1])

    loss = model.bpr_loss(*model(users, pos, neg))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.user_embedding.weight.grad is not None
    assert torch.isfinite(model.user_embedding.weight.grad).all()


@pytest.mark.parametrize("name", sorted(MODELS))
def test_predict_batch_matches_per_user_predict(name: str) -> None:
    model = _model(name).eval()
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        single = model.predict(0, items)
        batched = model.predict_batch(torch.tensor([0, 1]), items)

    assert single.shape == (N_ITEMS,)
    assert batched.shape == (2, N_ITEMS)
    assert torch.allclose(batched[0], single, atol=1e-6)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_predict_is_deterministic_in_eval_mode(name: str) -> None:
    model = _model(name).eval()
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        first = model.predict(3, items)
        second = model.predict(3, items)

    assert torch.equal(first, second)


@pytest.mark.parametrize("name", ["vbpr", "avbpr", "deepstyle", "vnpr"])
def test_visual_models_reject_missing_embeddings(name: str) -> None:
    cls, config, _ = MODELS[name]

    with pytest.raises((AssertionError, RuntimeError)):
        cls(N_USERS, N_ITEMS, visual_embeddings=None, config=config)
