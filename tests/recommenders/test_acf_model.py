"""Unit tests for the ACF recommender (Chen et al., SIGIR 2017).

Covers the paper's score (Eq. 6), the regularisation groups (Eq. 5),
the SVD++ degeneration under uniform attention (Section 4.1) and the
seeded uniform history sub-sampling used when ``|R(u)| > max_history``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recommenders.acf import ACF

N_USERS = 6
N_ITEMS = 10
N_COMPONENTS = 7
RAW_DIM = 8
FULL_HISTORY_USER = 3  # interacts with every item {0..9}


def _components() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_ITEMS, N_COMPONENTS, RAW_DIM)).astype("float32")


def _history() -> dict[int, set[int]]:
    return {0: {1, 2, 3}, 1: {4, 5}, 2: set(), FULL_HISTORY_USER: set(range(N_ITEMS))}


def _config(**overrides: object) -> dict:
    base = {"latent_dim": 4, "visual_dim": 5, "att_hidden": 6, "max_history": 4, "l2_reg": 1e-4}
    return {**base, **overrides}


def _model(**overrides: object) -> ACF:
    torch.manual_seed(0)
    return ACF(
        N_USERS,
        N_ITEMS,
        visual_embeddings=_components(),
        config=_config(**overrides),
        train_interactions=_history(),
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.tensor([0, 1, 2, 3]), torch.tensor([1, 4, 7, 9]), torch.tensor([8, 9, 0, 1])


def _force_uniform_item_attention(model: ACF) -> None:
    """Zero the energy head so ``a(u,l)`` is constant and ``α`` uniform on the mask."""
    with torch.no_grad():
        model.item_attention.score.weight.zero_()
        model.item_attention.score.bias.zero_()


def _reset_eval_cache(model: ACF) -> None:
    model.train()
    model.eval()


# ------------------------------------------------------------------ basics
def test_forward_returns_two_score_vectors() -> None:
    model = _model()
    users, pos, neg = _batch()

    score_pos, score_neg = model(users, pos, neg)

    assert score_pos.shape == (4,)
    assert score_neg.shape == (4,)


def test_bpr_loss_is_finite_and_backpropagates() -> None:
    model = _model()
    users, pos, neg = _batch()

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


# ------------------------------------------------------- score (Eq. 6, §4.1)
@pytest.mark.parametrize("user", [0, FULL_HISTORY_USER])
def test_uniform_item_attention_degenerates_to_svdpp(user: int) -> None:
    model = _model().eval()
    _force_uniform_item_attention(model)
    items = torch.arange(N_ITEMS)
    history = model.history_items[user][model.history_mask[user]]

    with torch.no_grad():
        scores = model.predict(user, items)
        gamma_u = model.user_embedding.weight[user]
        p_hat = gamma_u + model.aux_embedding.weight[history].mean(dim=0)
        expected = model.item_embedding.weight @ p_hat

    assert history.numel() > 0
    assert torch.allclose(scores, expected, atol=1e-6)


def test_uniform_item_attention_ignores_visual_and_item_latent_of_history() -> None:
    model = _model().eval()
    _force_uniform_item_attention(model)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        before = model.predict(0, items)
        model.visual_features[1] += 5.0  # item 1 ∈ R(0): only feeds the energy
        model.item_embedding.weight[2] *= 3.0  # item 2 ∈ R(0): idem, v_l not aggregated
        _reset_eval_cache(model)
        after = model.predict(0, items)

    assert torch.allclose(before[[0, 3, 4, 5, 6, 7, 8, 9]], after[[0, 3, 4, 5, 6, 7, 8, 9]])


def test_model_has_no_item_bias() -> None:
    model = _model()

    assert not hasattr(model, "item_bias")
    assert all("item_bias" not in name for name, _ in model.named_parameters())


def test_candidate_score_does_not_depend_on_its_own_components() -> None:
    model = _model().eval()
    items = torch.arange(N_ITEMS)
    outside_history = 7  # not in R(0) = {1, 2, 3}

    with torch.no_grad():
        before = model.predict(0, items)
        model.visual_features[outside_history] += 10.0
        _reset_eval_cache(model)
        after = model.predict(0, items)

    assert torch.equal(before, after)


def test_history_item_components_do_change_the_profile() -> None:
    model = _model().eval()
    items = torch.arange(N_ITEMS)
    inside_history = 1  # in R(0)

    with torch.no_grad():
        before = model.predict(0, items)
        model.visual_features[inside_history] += 10.0
        _reset_eval_cache(model)
        after = model.predict(0, items)

    assert not torch.allclose(before, after)


def test_forward_predict_and_predict_batch_share_one_formula() -> None:
    model = _model().eval()
    users, pos, neg = _batch()

    with torch.no_grad():
        score_pos, score_neg = model(users, pos, neg)
        per_user_pos = torch.stack(
            [model.predict(int(u), pos[i : i + 1])[0] for i, u in enumerate(users)]
        )
        batched = model.predict_batch(users, torch.arange(N_ITEMS))

    assert torch.allclose(score_pos, per_user_pos, atol=1e-6)
    assert torch.allclose(score_pos, batched[torch.arange(4), pos], atol=1e-6)
    assert torch.allclose(score_neg, batched[torch.arange(4), neg], atol=1e-6)


# ------------------------------------------------------ regularisation (Eq. 5)
def _l2_after_forward(model: ACF) -> torch.Tensor:
    users, pos, neg = _batch()
    model(users, pos, neg)
    return model.l2_reg().detach()


def test_attention_nets_and_projections_are_not_regularised() -> None:
    model = _model()
    reference = _l2_after_forward(model)
    theta = (
        model.component_attention,
        model.item_attention,
        model.comp_projection,
        model.visual_to_latent,
    )

    with torch.no_grad():
        for module in theta:
            for param in module.parameters():
                param.mul_(10.0)
    scaled = _l2_after_forward(model)

    assert reference > 0
    assert torch.allclose(scaled, reference)
    assert model._l2_shared_terms() == []


def test_history_aux_rows_are_regularised() -> None:
    model = _model()
    reference = _l2_after_forward(model)
    history_item = 5  # in R(1), never a pos/neg of the batch

    with torch.no_grad():
        model.aux_embedding.weight[history_item] *= 10.0
    scaled = _l2_after_forward(model)

    assert scaled > reference


def test_aux_rows_outside_batch_histories_are_not_regularised() -> None:
    model = _model()
    users, pos, neg = torch.tensor([0, 1]), torch.tensor([1, 4]), torch.tensor([8, 9])
    untouched_item = 6  # not in R(0) ∪ R(1) = {1..5}, nor a pos/neg
    model(users, pos, neg)
    reference = model.l2_reg().detach()

    with torch.no_grad():
        model.aux_embedding.weight[untouched_item] *= 10.0
    model(users, pos, neg)
    scaled = model.l2_reg().detach()

    assert torch.allclose(scaled, reference)


# ------------------------------------------------------------- history buffer
def test_history_buffer_pads_and_masks() -> None:
    model = _model()

    assert model.history_items.shape == (N_USERS, 4)
    assert model.history_mask[0].sum().item() == 3
    assert set(model.history_items[0][model.history_mask[0]].tolist()) == {1, 2, 3}
    assert model.history_mask[2].sum().item() == 0


def test_history_over_bound_is_uniformly_subsampled_not_an_ordered_prefix() -> None:
    horizon = 4
    seeds = range(20)

    chosen = [
        set(_model(history_seed=seed).history_items[FULL_HISTORY_USER][:horizon].tolist())
        for seed in seeds
    ]
    mean_index = float(np.mean([np.mean(sorted(s)) for s in chosen]))

    assert all(len(s) == horizon and s <= set(range(N_ITEMS)) for s in chosen)
    assert any(s != {0, 1, 2, 3} for s in chosen)
    # Uniform sampling from {0..9} has E[index] = 4.5; the old prefix gave 1.5.
    assert mean_index > 3.0


def test_history_subsampling_is_deterministic_given_the_seed() -> None:
    first = _model(history_seed=7).history_items[FULL_HISTORY_USER].tolist()
    second = _model(history_seed=7).history_items[FULL_HISTORY_USER].tolist()
    other = _model(history_seed=8).history_items[FULL_HISTORY_USER].tolist()

    assert first == second
    assert set(first) != set(other) or first != other


def test_history_seed_defaults_to_42() -> None:
    default = _model().history_items[FULL_HISTORY_USER].tolist()
    explicit = _model(history_seed=42).history_items[FULL_HISTORY_USER].tolist()

    assert default == explicit


def test_max_history_none_keeps_the_complete_history() -> None:
    model = _model(max_history=None)

    assert model.max_history is None
    assert model.history_items.shape == (N_USERS, N_ITEMS)  # H = longest history
    assert model.history_mask[FULL_HISTORY_USER].all()
    assert set(model.history_items[FULL_HISTORY_USER].tolist()) == set(range(N_ITEMS))
    assert model.history_mask[0].sum().item() == 3
