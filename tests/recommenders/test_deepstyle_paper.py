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


class TestCostParity:
    def test_projection_param_count_matches_single_linear_layer(self) -> None:
        # Linear E has exactly VBPR's projection size; the only extra
        # visual-path params are the category table (n_cat × d).
        ds = _deepstyle(_categories())
        e_params = sum(p.numel() for p in ds.visual_projection.parameters())

        assert e_params == RAW_DIM * K  # single linear, no hidden layer
