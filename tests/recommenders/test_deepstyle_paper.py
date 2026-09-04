"""DeepStyle (paper formulation) contract tests.

Pins the properties of Liu, Wu & Wang (SIGIR 2017), Eqs. 2-3 and 6:
1. the visual projection E is LINEAR (not an MLP) and lands in R^d;
2. the category vector l is a LEARNED embedding shared per category,
   subtracted in the projected space;
3. one user vector p_u, one dimension d, no item bias;
4. a single λ over every parameter (Eq. 6);
5. without category labels the model analytically degenerates to a
   RESTRICTED VBPR (γ_u ≡ θ_u, k_v = k, no β_i, no β'): the null
   category term p_u^T l is constant per user, so pairwise differences
   and rankings match exactly under shared weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.recommenders.deepstyle import DeepStyle
from src.recommenders.vbpr import VBPR

N_USERS, N_ITEMS, RAW_DIM, K = 6, 40, 16, 5
EXPECTED_STATE_KEYS = {
    "user_embedding.weight",
    "item_embedding.weight",
    "visual_projection.weight",
    "category_embedding.weight",
}


def _visual() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((N_ITEMS, RAW_DIM)).astype("float32")


def _categories() -> np.ndarray:
    return (np.arange(N_ITEMS) % 4).astype(np.int64)  # 4 categories


def _deepstyle(cats: np.ndarray | None, config: dict | None = None) -> DeepStyle:
    torch.manual_seed(3)
    return DeepStyle(
        N_USERS,
        N_ITEMS,
        visual_embeddings=_visual(),
        config=config or {"latent_dim": K, "l2_reg": 1e-4},
        item_categories=cats,
    )


class TestPaperFormulation:
    def test_projection_is_linear_and_lands_in_latent_dim(self) -> None:
        # Arrange / Act
        model = _deepstyle(_categories())

        # Assert
        assert isinstance(model.visual_projection, nn.Linear)
        assert model.visual_projection.in_features == RAW_DIM
        assert model.visual_projection.out_features == model.config["latent_dim"]
        assert model.visual_projection.bias is None
        assert not any(isinstance(m, nn.ReLU) for m in model.modules())

    def test_category_embedding_is_learned_and_shared_per_category(self) -> None:
        # Arrange
        model = _deepstyle(_categories())
        items = torch.tensor([0, 4])  # both category 0

        # Act
        with torch.no_grad():
            projected = model.visual_projection(model._resolve_visual(items))
            style = model._item_visual_term(items)
        subtracted = projected - style

        # Assert
        assert isinstance(model.category_embedding, nn.Embedding)
        assert model.category_embedding.num_embeddings == 4
        assert model.category_embedding.embedding_dim == K
        assert model.category_embedding.weight.requires_grad
        assert torch.allclose(subtracted[0], subtracted[1])

    def test_category_is_subtracted_in_projected_space(self) -> None:
        # Arrange
        model = _deepstyle(_categories())
        items = torch.arange(N_ITEMS)

        # Act
        with torch.no_grad():
            style = model._item_visual_term(items)
            expected = model.visual_projection(
                model._resolve_visual(items)
            ) - model.category_embedding(model.item_category_idx[items])

        # Assert
        assert style.shape == (N_ITEMS, K)
        assert torch.equal(style, expected)

    def test_bpr_loss_backpropagates_into_category_embedding(self) -> None:
        # Arrange
        model = _deepstyle(_categories())

        # Act
        loss = model.bpr_loss(
            *model(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 8]))
        )
        loss.backward()

        # Assert
        assert model.category_embedding.weight.grad is not None
        assert model.category_embedding.weight.grad.abs().sum() > 0

    def test_score_is_single_user_vector_dotted_with_style_plus_latent(self) -> None:
        # Arrange
        model = _deepstyle(_categories()).eval()
        user = 2
        items = torch.arange(N_ITEMS)

        # Act
        with torch.no_grad():
            predicted = model.predict(user, items)
            p_u = model.user_embedding.weight[user]
            e_f = model.visual_projection(model._resolve_visual(items))
            l_cat = model.category_embedding(model.item_category_idx[items])
            q_i = model.item_embedding(items)
            expected = ((e_f - l_cat + q_i) * p_u).sum(-1)

        # Assert
        assert torch.allclose(predicted, expected, atol=1e-6)

    def test_wants_categories_flag(self) -> None:
        assert DeepStyle.wants_categories is True

    def test_wrong_category_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            _deepstyle(np.zeros(3, dtype=np.int64))


class TestSingleDimensionNoBias:
    def test_style_dim_different_from_latent_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="style_dim"):
            _deepstyle(_categories(), {"latent_dim": K, "style_dim": K + 1, "l2_reg": 1e-4})

    def test_style_dim_equal_to_latent_dim_is_accepted(self) -> None:
        model = _deepstyle(_categories(), {"latent_dim": K, "style_dim": K, "l2_reg": 1e-4})

        assert model.visual_projection.out_features == K

    def test_no_item_bias_and_no_separate_style_user_table(self) -> None:
        # Arrange / Act
        model = _deepstyle(_categories())

        # Assert
        assert not hasattr(model, "item_bias")
        assert not hasattr(model, "style_user_embedding")
        assert model._visual_user_table() is model.user_embedding
        assert set(model.state_dict().keys()) == EXPECTED_STATE_KEYS

    def test_every_parameter_is_regularised_under_the_single_lambda(self) -> None:
        # Arrange
        model = _deepstyle(_categories())
        users, pos, neg = torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 8])

        # Act
        shared_keys = {key for key, _ in model._l2_shared_terms()}
        gathered = model._l2_gathered_terms(users, pos, neg)
        gathered_keys = {key for key, _ in gathered}

        # Assert
        assert shared_keys == {"l2_reg"}
        assert gathered_keys == {"l2_reg"}
        # p_u, q_i(pos), q_i(neg), l(pos), l(neg): the category rows are gathered.
        assert len(gathered) == 5
        assert all(rows.shape[-1] == K for _, rows in gathered)


class TestTradesyDegeneration:
    """No categories → analytic equivalence with a RESTRICTED VBPR.

    The restricted VBPR ties its visual user weights to the latent ones
    (θ_u ≡ γ_u), uses k_v = k, and zeroes β_i and (when present) β'.
    """

    def _paired_models(self) -> tuple[VBPR, DeepStyle]:
        torch.manual_seed(3)
        vbpr = VBPR(
            N_USERS,
            N_ITEMS,
            visual_embeddings=_visual(),
            config={"latent_dim": K, "visual_dim": K, "l2_reg": 1e-4},
        )
        ds = _deepstyle(None)  # no categories -> single null category
        # Copy DeepStyle's weights into the VBPR and restrict it: one
        # user vector for both tables, no item bias, no visual bias.
        with torch.no_grad():
            vbpr.user_embedding.weight.copy_(ds.user_embedding.weight)
            vbpr.visual_user_embedding.weight.copy_(ds.user_embedding.weight)
            vbpr.item_embedding.weight.copy_(ds.item_embedding.weight)
            vbpr.visual_projection.weight.copy_(ds.visual_projection.weight)
            vbpr.item_bias.weight.zero_()
            visual_bias = getattr(vbpr, "visual_bias", None)
            if visual_bias is not None:
                visual_bias_weight = getattr(visual_bias, "weight", visual_bias)
                visual_bias_weight.zero_()
        return vbpr.eval(), ds.eval()

    def test_single_null_category(self) -> None:
        ds = _deepstyle(None)

        assert ds.n_categories == 1
        assert ds.item_category_idx.unique().tolist() == [0]

    def test_pairwise_differences_match_restricted_vbpr(self) -> None:
        # Arrange
        vbpr, ds = self._paired_models()
        users = torch.tensor([0, 1, 2])
        pos = torch.tensor([1, 5, 9])
        neg = torch.tensor([3, 7, 11])

        # Act
        with torch.no_grad():
            v_pos, v_neg = vbpr(users, pos, neg)
            d_pos, d_neg = ds(users, pos, neg)

        # Assert — the constant p_u·l cancels in the pairwise difference,
        # the quantity BPR optimises.  This is the degeneration, exactly.
        assert torch.allclose(v_pos - v_neg, d_pos - d_neg, atol=1e-6)

    def test_rankings_match_restricted_vbpr(self) -> None:
        # Arrange
        vbpr, ds = self._paired_models()
        items = torch.arange(N_ITEMS)

        # Act / Assert
        with torch.no_grad():
            for user in range(N_USERS):
                rank_v = torch.sort(vbpr.predict(user, items), descending=True, stable=True).indices
                rank_d = torch.sort(ds.predict(user, items), descending=True, stable=True).indices
                assert torch.equal(rank_v, rank_d)
            batch_v = vbpr.predict_batch(torch.arange(N_USERS), items)
            batch_d = ds.predict_batch(torch.arange(N_USERS), items)
        assert torch.equal(
            torch.argsort(batch_v, dim=1, descending=True, stable=True),
            torch.argsort(batch_d, dim=1, descending=True, stable=True),
        )


class TestCostParity:
    def test_projection_param_count_matches_single_linear_layer(self) -> None:
        # Linear E has exactly VBPR's projection size; the only extra
        # visual-path params are the category table (n_cat × d).
        ds = _deepstyle(_categories())
        e_params = sum(p.numel() for p in ds.visual_projection.parameters())

        assert e_params == RAW_DIM * K  # single linear, no hidden layer


# ---------------------------------------------------------------- training
class _RestrictedVBPR(VBPR):
    """The RESTRICTED VBPR of the module docstring, kept restricted under training.

    Exactly the restrictions the degeneration is stated against, and
    nothing else: one user table (``θ_u ≡ γ_u``, so the gathered L2 sees
    ``γ_u`` once, as DeepStyle sees ``p_u``), ``β_i ≡ 0`` and ``β' ≡ 0``
    frozen.  Scoring, loss and every other parameter are VBPR's own.
    """

    _L2_USER_TABLES = ("user_embedding",)

    def _visual_user_table(self) -> nn.Embedding:
        return self.user_embedding


def _synthetic_interactions(seed: int = 5) -> dict[int, set[int]]:
    rng = np.random.default_rng(seed)
    return {u: set(rng.choice(N_ITEMS, size=6, replace=False).tolist()) for u in range(N_USERS)}


def _trained_pair(l2_reg: float, steps: int = 30) -> tuple[_RestrictedVBPR, DeepStyle]:
    """Same init, same batches, same optimiser: ``steps`` real BPR steps each."""
    from src.utils.amp_compat import get_grad_scaler
    from src.utils.training import BPRBatchSampler, bpr_step

    ds = _deepstyle(None, {"latent_dim": K, "l2_reg": l2_reg})
    torch.manual_seed(3)
    vbpr = _RestrictedVBPR(
        N_USERS,
        N_ITEMS,
        visual_embeddings=_visual(),
        # DeepStyle puts E under its single λ (Eq. 6); VBPR's λ_E defaults to 0.
        config={"latent_dim": K, "visual_dim": K, "l2_reg": l2_reg, "l2_reg_projection": l2_reg},
    )
    with torch.no_grad():
        vbpr.user_embedding.weight.copy_(ds.user_embedding.weight)
        vbpr.item_embedding.weight.copy_(ds.item_embedding.weight)
        vbpr.visual_projection.weight.copy_(ds.visual_projection.weight)
        vbpr.visual_user_embedding.weight.zero_()  # unreachable under the tie
        vbpr.item_bias.weight.zero_()
        vbpr.visual_bias.zero_()
    for frozen in (vbpr.visual_user_embedding.weight, vbpr.item_bias.weight, vbpr.visual_bias):
        frozen.requires_grad_(False)
    # float64 so the only tolerance left is the analytic cancellation itself.
    vbpr.double().train()
    ds.double().train()

    sampler = BPRBatchSampler(_synthetic_interactions(), N_ITEMS, batch_size=8, seed=11)
    batches = [batch for epoch in range(steps) for batch in sampler.epoch(epoch)][:steps]
    assert len(batches) == steps
    optimisers = (
        torch.optim.SGD([p for p in vbpr.parameters() if p.requires_grad], lr=0.1),
        torch.optim.SGD([p for p in ds.parameters() if p.requires_grad], lr=0.1),
    )
    scaler = get_grad_scaler(enabled=False)
    for users, pos, neg in batches:
        for model, optimiser in zip((vbpr, ds), optimisers, strict=True):
            bpr_step(model, optimiser, scaler, users, pos, neg, device="cpu", use_cuda=False)
    return vbpr.eval(), ds.eval()


@pytest.mark.parametrize("l2_reg", [0.0, 1e-3], ids=["no_l2", "l2"])
class TestTradesyDegenerationAfterTraining:
    """The degeneration is a property of the OBJECTIVE, so it survives training.

    ``∂(ŷ_ui − ŷ_uj)/∂l = −p_u + p_u = 0`` and ``l`` cancels in the
    gradients of ``p_u``, ``q_i`` and ``E``: the shared parameters of
    the two models follow the same trajectory and only ``l`` drifts
    (under L2).  Neither the pairwise scores nor the per-user rankings
    can therefore diverge.
    """

    def test_training_actually_moves_the_shared_parameters(self, l2_reg: float) -> None:
        vbpr, ds = _trained_pair(l2_reg)
        fresh = _deepstyle(None, {"latent_dim": K, "l2_reg": l2_reg}).double()

        assert not torch.allclose(ds.user_embedding.weight, fresh.user_embedding.weight)
        assert not torch.allclose(ds.item_embedding.weight, fresh.item_embedding.weight)
        assert not torch.allclose(vbpr.visual_projection.weight, fresh.visual_projection.weight)

    def test_shared_parameters_follow_the_same_trajectory(self, l2_reg: float) -> None:
        vbpr, ds = _trained_pair(l2_reg)

        torch.testing.assert_close(vbpr.user_embedding.weight, ds.user_embedding.weight)
        torch.testing.assert_close(vbpr.item_embedding.weight, ds.item_embedding.weight)
        torch.testing.assert_close(vbpr.visual_projection.weight, ds.visual_projection.weight)
        assert torch.all(vbpr.item_bias.weight == 0) and torch.all(vbpr.visual_bias == 0)

    def test_pairwise_differences_match_after_training(self, l2_reg: float) -> None:
        vbpr, ds = _trained_pair(l2_reg)
        users, pos, neg = (
            torch.tensor([0, 1, 2, 5]),
            torch.tensor([1, 5, 9, 2]),
            torch.tensor([3, 7, 11, 30]),
        )

        with torch.no_grad():
            v_pos, v_neg = vbpr(users, pos, neg)
            d_pos, d_neg = ds(users, pos, neg)

        torch.testing.assert_close(v_pos - v_neg, d_pos - d_neg, rtol=0, atol=1e-9)

    def test_rankings_match_after_training_in_predict_and_predict_batch(
        self, l2_reg: float
    ) -> None:
        vbpr, ds = _trained_pair(l2_reg)
        items, users = torch.arange(N_ITEMS), torch.arange(N_USERS)

        with torch.no_grad():
            batch_v = vbpr.predict_batch(users, items)
            batch_d = ds.predict_batch(users, items)
            for user in range(N_USERS):
                rank_v = torch.argsort(vbpr.predict(user, items), descending=True, stable=True)
                rank_d = torch.argsort(ds.predict(user, items), descending=True, stable=True)
                assert torch.equal(rank_v, rank_d), f"user {user}"
        assert torch.equal(
            torch.argsort(batch_v, dim=1, descending=True, stable=True),
            torch.argsort(batch_d, dim=1, descending=True, stable=True),
        )

    def test_absolute_scores_differ_by_the_per_user_constant_p_u_dot_l(self, l2_reg: float) -> None:
        vbpr, ds = _trained_pair(l2_reg)
        items, users = torch.arange(N_ITEMS), torch.arange(N_USERS)

        with torch.no_grad():
            offset = ds.predict_batch(users, items) - vbpr.predict_batch(users, items)
            expected = -(ds.user_embedding.weight @ ds.category_embedding.weight[0])

        assert not torch.allclose(offset, torch.zeros_like(offset))  # l ≠ 0: scores differ
        torch.testing.assert_close(offset, expected.unsqueeze(1).expand_as(offset))


# ------------------------------------------------------ category cancellation
class TestNullCategoryCancellation:
    """Perturbing the single ``l`` moves absolute scores, never pairs or rankings."""

    def _scores(self, ds: DeepStyle) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        users, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([1, 5, 9]), torch.tensor([3, 7, 11])
        with torch.no_grad():
            s_pos, s_neg = ds(users, pos, neg)
            batch = ds.predict_batch(torch.arange(N_USERS), torch.arange(N_ITEMS))
            single = torch.stack([ds.predict(u, torch.arange(N_ITEMS)) for u in range(N_USERS)])
        return s_pos - s_neg, batch, single

    def _perturbed(self) -> tuple[DeepStyle, torch.Tensor]:
        ds = _deepstyle(None).eval()
        torch.manual_seed(9)
        delta = torch.randn(K)
        with torch.no_grad():
            ds.category_embedding.weight[0] += delta
        ds.train().eval()  # drop the full-catalogue style cache
        return ds, delta

    def test_pairwise_difference_and_rankings_are_invariant(self) -> None:
        before = self._scores(_deepstyle(None).eval())
        ds, _ = self._perturbed()

        after = self._scores(ds)

        torch.testing.assert_close(after[0], before[0])
        for name, b, a in (
            ("predict_batch", before[1], after[1]),
            ("predict", before[2], after[2]),
        ):
            assert torch.equal(
                torch.argsort(b, dim=1, descending=True, stable=True),
                torch.argsort(a, dim=1, descending=True, stable=True),
            ), name

    def test_absolute_scores_shift_by_minus_p_u_dot_delta(self) -> None:
        before = self._scores(_deepstyle(None).eval())
        ds, delta = self._perturbed()

        after = self._scores(ds)
        shift = -(ds.user_embedding.weight @ delta).unsqueeze(1)

        assert not torch.allclose(after[1], before[1])
        torch.testing.assert_close(after[1] - before[1], shift.expand_as(before[1]))
        torch.testing.assert_close(after[2] - before[2], shift.expand_as(before[2]))

    def test_with_real_categories_the_cancellation_does_not_hold(self) -> None:
        """Negative control: the cancellation is a single-category property."""
        ds = _deepstyle(_categories()).eval()
        before = self._scores(ds)[0]
        with torch.no_grad():
            ds.category_embedding.weight[0] += 1.0  # pos item 1 (cat 1) vs neg item 3 (cat 3)
            ds.category_embedding.weight[1] += 1.0  # cat(1)=1 shifts, cat(3)=3 does not
        ds.train().eval()

        after = self._scores(ds)[0]

        assert not torch.allclose(after, before)
