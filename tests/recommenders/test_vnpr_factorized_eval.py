"""VNPR factored predict_batch — declared NOT bit-identical, tolerance-checked.

The first MLP layer is factored (user half + precomputed item half);
this is mathematically equivalent but reorders float reductions.  The
acceptance criteria are: scores within fp32 tolerance, any ranking
swaps confined to sub-tolerance score gaps, and ranking METRICS
unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_all_metrics
from src.recommenders.vnpr import VNPR

N_USERS, N_ITEMS, DV = 30, 200, 24


@pytest.fixture()
def model() -> VNPR:
    torch.manual_seed(0)
    visual = np.random.default_rng(0).standard_normal((N_ITEMS, DV)).astype("float32")
    return VNPR(
        N_USERS,
        N_ITEMS,
        visual_embeddings=visual,
        config={"latent_dim": 6, "hidden_layers": [12, 6], "l2_reg": 1e-4},
    ).eval()


def _reference(model: VNPR, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    """Unfactored path: per-user predict() through _score (concat + full MLP)."""
    return torch.stack([model.predict(int(u), items) for u in users])


def test_scores_within_fp32_tolerance(model: VNPR) -> None:
    users = torch.arange(N_USERS)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = _reference(model, users, items)
        out = model.predict_batch(users, items)

    assert out.shape == (N_USERS, N_ITEMS)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)


def test_ranking_swaps_only_at_subtolerance_gaps(model: VNPR) -> None:
    users = torch.arange(N_USERS)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = _reference(model, users, items)
        out = model.predict_batch(users, items)

    r_ref = torch.sort(ref, dim=1, descending=True, stable=True).indices
    r_out = torch.sort(out, dim=1, descending=True, stable=True).indices
    swaps = (r_ref != r_out).nonzero()
    for u, p in swaps.tolist():
        gap = abs(ref[u, r_ref[u, p]] - ref[u, r_out[u, p]]).item()
        assert gap < 1e-6, f"ranking swap at a non-tie gap ({gap:.2e})"


def test_ranking_metrics_unchanged(model: VNPR) -> None:
    users = torch.arange(N_USERS)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = _reference(model, users, items)
        out = model.predict_batch(users, items)

    r_ref = torch.sort(ref, dim=1, descending=True, stable=True).indices
    r_out = torch.sort(out, dim=1, descending=True, stable=True).indices
    rng = np.random.default_rng(1)
    for u in range(N_USERS):
        gt = {int(x) for x in rng.choice(N_ITEMS, 3, replace=False)}
        m_ref = compute_all_metrics(r_ref[u].tolist()[:20], gt, [5, 10, 20])
        m_out = compute_all_metrics(r_out[u].tolist()[:20], gt, [5, 10, 20])
        assert m_ref == m_out


def test_item_first_layer_cache_invalidated_by_train(model: VNPR) -> None:
    users = torch.arange(2)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        model.predict_batch(users, items)
    assert model._item_first_layer_cache is not None

    model.train()
    assert model._item_first_layer_cache is None


def test_item_subset_not_cached_but_correct(model: VNPR) -> None:
    users = torch.arange(3)
    items = torch.tensor([0, 7, 42, 199])

    with torch.no_grad():
        ref = _reference(model, users, items)
        out = model.predict_batch(users, items)

    assert model._item_first_layer_cache is None  # subset must not populate cache
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)
