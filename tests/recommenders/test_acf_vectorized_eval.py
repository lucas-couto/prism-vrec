"""ACF vectorized predict_batch must match the per-user reference path.

``predict_batch`` computes each user's profile ``p̂_u`` once and scores
the candidates with a single GEMM ``p̂ @ V^T`` (Eq. 6 has no
per-candidate attention term).  GEMMs may reorder float reductions, so
equality is asserted at fp32 noise level (1e-5) AND — the property the
metrics actually depend on — the stable-sort rankings must be identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recommenders.acf import ACF

N_USERS, N_ITEMS, M, RAW_DIM = 24, 150, 5, 12


@pytest.fixture()
def model() -> ACF:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    comps = rng.standard_normal((N_ITEMS, M, RAW_DIM)).astype("float32")
    hist = {
        u: set(rng.choice(N_ITEMS, size=int(rng.integers(1, 20)), replace=False).tolist())
        for u in range(N_USERS)
    }
    hist[3] = set()  # empty-history user
    cfg = {"latent_dim": 6, "visual_dim": 8, "att_hidden": 7, "max_history": 10, "l2_reg": 1e-4}
    return ACF(
        N_USERS, N_ITEMS, visual_embeddings=comps, config=cfg, train_interactions=hist
    ).eval()


def _reference(model: ACF, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    """The pre-vectorization semantics: one predict() per user."""
    return torch.stack([model.predict(int(u), items) for u in users])


def test_scores_match_reference_within_fp32_noise(model: ACF) -> None:
    users = torch.arange(N_USERS)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = _reference(model, users, items)
        model.train()
        model.eval()  # reset caches between paths
        out = model.predict_batch(users, items)

    assert out.shape == (N_USERS, N_ITEMS)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)


def test_rankings_identical_to_reference(model: ACF) -> None:
    users = torch.arange(N_USERS)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = _reference(model, users, items)
        model.train()
        model.eval()
        out = model.predict_batch(users, items)

    ref_rank = torch.sort(ref, dim=1, descending=True, stable=True).indices
    out_rank = torch.sort(out, dim=1, descending=True, stable=True).indices
    assert torch.equal(ref_rank, out_rank)


def test_predict_batch_equals_profile_times_item_table(model: ACF) -> None:
    users = torch.arange(N_USERS)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        out = model.predict_batch(users, items)
        gamma_u = model.user_embedding(users)
        p_hat = model._augmented_user(users, gamma_u)
        expected = p_hat @ model.item_embedding.weight.T

    assert torch.allclose(out, expected, atol=1e-6)


def test_comp_cache_is_built_in_eval_and_invalidated_by_train(model: ACF) -> None:
    users = torch.arange(2)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        model.predict_batch(users, items)
    assert model._comp_cache is not None
    assert model._comp_cache.shape[0] == N_ITEMS

    model.train()
    assert model._comp_cache is None


def test_training_mode_does_not_use_the_cache(model: ACF) -> None:
    model.train()
    users, pos, neg = torch.arange(4), torch.arange(4), torch.arange(4, 8)

    model(users, pos, neg)

    assert model._comp_cache is None


def test_subset_of_items_works(model: ACF) -> None:
    users = torch.arange(4)
    items = torch.tensor([5, 17, 42, 99])

    with torch.no_grad():
        ref = _reference(model, users, items)
        out = model.predict_batch(users, items)

    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)
