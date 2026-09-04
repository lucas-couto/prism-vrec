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


# ------------------------------------------- SVD++ degeneration under training
def _freeze_uniform_item_attention(model: ACF) -> None:
    """Zero AND freeze the energy head: ``a(u,l)`` stays constant through training."""
    _force_uniform_item_attention(model)
    model.item_attention.score.weight.requires_grad_(False)
    model.item_attention.score.bias.requires_grad_(False)


def _svdpp_scores(model: ACF, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    """Explicit ``(γ_u + |R(u)|⁻¹ Σ_{l∈R(u)} p_l) · γ_j`` over the model's own ``R(u)``."""
    rows = []
    for user in users.tolist():
        history = model.history_items[user][model.history_mask[user]]
        aux = model.aux_embedding.weight[history]
        pooled = aux.mean(dim=0) if history.numel() else torch.zeros_like(aux.sum(dim=0))
        rows.append(
            model.item_embedding.weight[items] @ (model.user_embedding.weight[user] + pooled)
        )
    return torch.stack(rows)


def _sampleable_history() -> dict[int, set[int]]:
    """Users with at least one negative: the full-history user has none to draw."""
    return {u: s for u, s in _history().items() if len(s) < N_ITEMS}


def _train_uniform_acf(steps: int = 25) -> ACF:
    from src.utils.amp_compat import get_grad_scaler
    from src.utils.training import BPRBatchSampler, bpr_step

    model = _model()
    _freeze_uniform_item_attention(model)
    model.train()
    sampler = BPRBatchSampler(_sampleable_history(), N_ITEMS, batch_size=4, seed=13)
    batches = [batch for epoch in range(steps) for batch in sampler.epoch(epoch)][:steps]
    assert len(batches) == steps
    optimiser = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    scaler = get_grad_scaler(enabled=False)
    for users, pos, neg in batches:
        bpr_step(model, optimiser, scaler, users, pos, neg, device="cpu", use_cuda=False)
    return model.eval()


def test_frozen_uniform_attention_keeps_alpha_uniform_after_training() -> None:
    model = _train_uniform_acf()
    fresh = _model()

    assert torch.all(model.item_attention.score.weight == 0)
    assert torch.all(model.item_attention.score.bias == 0)
    # Real training: U, V and P moved.
    assert not torch.allclose(model.user_embedding.weight, fresh.user_embedding.weight)
    assert not torch.allclose(model.item_embedding.weight, fresh.item_embedding.weight)
    assert not torch.allclose(model.aux_embedding.weight, fresh.aux_embedding.weight)


def test_svdpp_degeneration_holds_after_training_in_every_scoring_path() -> None:
    model = _train_uniform_acf()
    users, items = torch.arange(N_USERS), torch.arange(N_ITEMS)
    expected = _svdpp_scores(model, users, items)  # (n_users, n_items)
    b_users, pos, neg = _batch()

    with torch.no_grad():
        batched = model.predict_batch(users, items)
        single = torch.stack([model.predict(u, items) for u in range(N_USERS)])
        score_pos, score_neg = model(b_users, pos, neg)

    assert torch.allclose(batched, expected, atol=1e-6)
    assert torch.allclose(single, expected, atol=1e-6)
    assert torch.allclose(score_pos, expected[b_users, pos], atol=1e-6)
    assert torch.allclose(score_neg, expected[b_users, neg], atol=1e-6)


def test_svdpp_degeneration_holds_in_train_mode_too() -> None:
    """The formula is not an artefact of the eval-mode component cache."""
    model = _train_uniform_acf().train()
    users, items = torch.arange(N_USERS), torch.arange(N_ITEMS)

    with torch.no_grad():
        batched = model.predict_batch(users, items)

    assert torch.allclose(batched, _svdpp_scores(model, users, items), atol=1e-6)


def test_unfrozen_energy_head_breaks_uniformity_under_training() -> None:
    """Negative control: zeroing alone is NOT enough — training moves ``w_1``/``c_1``."""
    from src.utils.amp_compat import get_grad_scaler
    from src.utils.training import BPRBatchSampler, bpr_step

    model = _model()
    _force_uniform_item_attention(model)  # zeroed, not frozen
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.05)
    scaler = get_grad_scaler(enabled=False)
    sampler = BPRBatchSampler(_sampleable_history(), N_ITEMS, batch_size=4, seed=13)
    for users, pos, neg in list(sampler.epoch(0))[:3]:
        bpr_step(model, optimiser, scaler, users, pos, neg, device="cpu", use_cuda=False)
    model.eval()

    assert not torch.all(model.item_attention.score.bias == 0) or not torch.all(
        model.item_attention.score.weight == 0
    )


def test_acf_consumes_fp16_components_and_projects_in_float32() -> None:
    """Pooled ``*_comp.npy`` are fp16; ACF must gather, cast and score them."""
    import numpy as np

    from src.recommenders.acf import ACF

    n_users, n_items = 6, 40
    visual = np.random.default_rng(1).standard_normal((n_items, 4, 6)).astype(np.float16)
    interactions = {u: {u, u + 1, u + 2} for u in range(n_users)}
    model = ACF(
        n_users,
        n_items,
        visual,
        {"latent_dim": 8, "att_hidden": 4, "max_history": 3, "history_seed": 0},
        train_interactions=interactions,
    )
    model._CACHE_CHUNK_ITEMS = 7  # exercise the chunked catalogue projection

    assert model.visual_features.dtype == torch.float16
    users = torch.arange(n_users)
    r_pos, r_neg = model(users, torch.arange(n_users), torch.arange(10, 10 + n_users))
    assert torch.isfinite(r_pos).all() and torch.isfinite(r_neg).all()
    assert model._projected_components(torch.tensor([0, 1])).dtype == torch.float32

    model.eval()
    scores = model.predict(0, torch.arange(n_items))
    assert scores.shape == (n_items,) and torch.isfinite(scores).all()
    assert model._comp_cache.shape[0] == n_items and model._comp_cache.dtype == torch.float32
