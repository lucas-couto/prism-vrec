"""DeepStyle (paper formulation) contract tests.

Pins the three properties the implementation must have:
1. the visual projection E is LINEAR (not an MLP), so cost ≈ VBPR;
2. the category vector is a LEARNED embedding shared per category,
   subtracted in the projected space;
3. without category labels the model analytically degenerates to VBPR
   (the null-category term cancels in the BPR pairwise difference and
   shifts every item of a user by the same constant — rankings and
   pairwise differences match VBPR exactly given shared weights).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.recommenders.deepstyle import DeepStyle
from src.recommenders.vbpr import VBPR

N_USERS, N_ITEMS, RAW_DIM, KS = 6, 40, 16, 5


def _visual() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((N_ITEMS, RAW_DIM)).astype("float32")


def _categories() -> np.ndarray:
    return (np.arange(N_ITEMS) % 4).astype(np.int64)  # 4 categories


def _deepstyle(cats: np.ndarray | None) -> DeepStyle:
    torch.manual_seed(3)
    return DeepStyle(
        N_USERS,
        N_ITEMS,
        visual_embeddings=_visual(),
        config={"latent_dim": 4, "style_dim": KS, "l2_reg": 1e-4},
        item_categories=cats,
    )


class TestPaperFormulation:
    def test_projection_is_linear_not_mlp(self) -> None:
        model = _deepstyle(_categories())

        assert isinstance(model.visual_projection, nn.Linear)
        assert model.visual_projection.in_features == RAW_DIM
        assert model.visual_projection.out_features == KS
        assert not any(isinstance(m, nn.ReLU) for m in model.modules())

    def test_category_embedding_learned_and_shared(self) -> None:
        cats = _categories()
        model = _deepstyle(cats)

        assert isinstance(model.category_embedding, nn.Embedding)
        assert model.category_embedding.num_embeddings == 4
        assert model.category_embedding.weight.requires_grad

        # Shared per category: two items of the same category subtract
        # the same vector.
        items = torch.tensor([0, 4])  # both category 0
        with torch.no_grad():
            f = model._resolve_visual(items)
            style = model._item_visual_term(items)
            projected = model.visual_projection(f)
        subtracted = projected - style
        assert torch.allclose(subtracted[0], subtracted[1])

    def test_subtraction_in_projected_space(self) -> None:
        model = _deepstyle(_categories())
        items = torch.arange(N_ITEMS)

        with torch.no_grad():
            style = model._item_visual_term(items)
            expected = model.visual_projection(
                model._resolve_visual(items)
            ) - model.category_embedding(model.item_category_idx[items])

        assert style.shape == (N_ITEMS, KS)
        assert torch.equal(style, expected)

    def test_category_gradient_flows(self) -> None:
        model = _deepstyle(_categories())
        loss = model.bpr_loss(
            *model(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 8]))
        )
        loss.backward()

        assert model.category_embedding.weight.grad is not None
        assert model.category_embedding.weight.grad.abs().sum() > 0

    def test_wants_categories_flag(self) -> None:
        assert DeepStyle.wants_categories is True

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            _deepstyle(np.zeros(3, dtype=np.int64))


class TestTradesyDegeneration:
    """No categories → analytic equivalence with VBPR under shared weights."""

    def _paired_models(self) -> tuple[VBPR, DeepStyle]:
        torch.manual_seed(3)
        vbpr = VBPR(
            N_USERS,
            N_ITEMS,
            visual_embeddings=_visual(),
            config={"latent_dim": 4, "visual_dim": KS, "l2_reg": 1e-4},
        )
        ds = _deepstyle(None)  # no categories -> single null category
        # Copy the shared weights so the analytic claim is testable exactly.
        with torch.no_grad():
            ds.user_embedding.weight.copy_(vbpr.user_embedding.weight)
            ds.item_embedding.weight.copy_(vbpr.item_embedding.weight)
            ds.item_bias.weight.copy_(vbpr.item_bias.weight)
            ds.style_user_embedding.weight.copy_(vbpr.visual_user_embedding.weight)
            ds.visual_projection.weight.copy_(vbpr.visual_projection.weight)
        return vbpr.eval(), ds.eval()

    def test_single_null_category(self) -> None:
        ds = _deepstyle(None)
        assert ds.n_categories == 1
        assert ds.item_category_idx.unique().tolist() == [0]

    def test_pairwise_differences_match_vbpr(self) -> None:
        vbpr, ds = self._paired_models()
        users = torch.tensor([0, 1, 2])
        pos = torch.tensor([1, 5, 9])
        neg = torch.tensor([3, 7, 11])

        with torch.no_grad():
            v_pos, v_neg = vbpr(users, pos, neg)
            d_pos, d_neg = ds(users, pos, neg)

        # The constant s_u·c0 cancels in the pairwise difference — the
        # quantity BPR optimises. This is the degeneration, exactly.
        assert torch.allclose(v_pos - v_neg, d_pos - d_neg, atol=1e-6)

    def test_rankings_match_vbpr(self) -> None:
        vbpr, ds = self._paired_models()
        items = torch.arange(N_ITEMS)

        with torch.no_grad():
            for user in range(N_USERS):
                rank_v = torch.sort(vbpr.predict(user, items), descending=True, stable=True).indices
                rank_d = torch.sort(ds.predict(user, items), descending=True, stable=True).indices
                assert torch.equal(rank_v, rank_d)


class TestCostParity:
    def test_visual_path_param_count_close_to_vbpr(self) -> None:
        # Linear E has exactly VBPR's projection size; the only extra
        # visual-path params are the category table (n_cat × ks).
        ds = _deepstyle(_categories())
        e_params = sum(p.numel() for p in ds.visual_projection.parameters())
        assert e_params == RAW_DIM * KS  # single linear, no hidden layer
