"""VNPR fidelity to Niu, Caverlee & Lu (WSDM 2018), Sections 4.1-4.3 and 5.2.

Every assertion is written against the paper's equations computed by
hand: mirrored branches with two item tables, element-wise product
merge, one-neuron ReLU dense, inference as the average of the two
branches, dropout on the embeddings and L2 over the WHOLE embedding
matrices (not BPR-Opt).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recommenders.vnpr import VNPR

N_USERS, N_ITEMS, K, DV = 30, 200, 6, 24


def _visual(seed: int = 0, shape: tuple[int, ...] = (N_ITEMS, DV)) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype("float32")


def _make(config: dict | None = None, visual: np.ndarray | None = None) -> VNPR:
    torch.manual_seed(0)
    cfg = {"latent_dim": K, "l2_reg": 1e-4, **(config or {})}
    return VNPR(
        N_USERS, N_ITEMS, visual_embeddings=_visual() if visual is None else visual, config=cfg
    )


@pytest.fixture()
def model() -> VNPR:
    return _make().eval()


def _manual_branch(
    model: VNPR, users: torch.Tensor, items: torch.Tensor, neg: bool
) -> torch.Tensor:
    """ReLU(w^T [p ∘ q, v ∘ f] + b) with q from W_i (neg=False) or W_i' (neg=True)."""
    table = model.item_embedding_neg if neg else model.item_embedding
    p = model.user_embedding.weight[users]
    q = table.weight[items]
    v = model.visual_user_embedding.weight[users]
    f = model.visual_features[items]
    merged = torch.cat([p * q, v * f], dim=-1)
    w = model.dense.weight.squeeze(0)
    return torch.relu(merged @ w + model.dense.bias)


def test_forward_matches_paper_branches_with_positive_and_negative_item_tables(model: VNPR) -> None:
    users = torch.arange(8)
    pos = torch.arange(10, 18)
    neg = torch.arange(50, 58)

    with torch.no_grad():
        r_pos, r_neg = model(users, pos, neg)

    assert torch.allclose(r_pos, _manual_branch(model, users, pos, neg=False), atol=1e-6)
    assert torch.allclose(r_neg, _manual_branch(model, users, neg, neg=True), atol=1e-6)
    assert not torch.allclose(r_neg, _manual_branch(model, users, neg, neg=False))


def test_predict_is_the_mean_of_both_branches_fed_the_same_item(model: VNPR) -> None:
    items = torch.arange(N_ITEMS)
    users = torch.full((N_ITEMS,), 3, dtype=torch.long)

    with torch.no_grad():
        out = model.predict(3, items)

    expected = 0.5 * (
        _manual_branch(model, users, items, neg=False)
        + _manual_branch(model, users, items, neg=True)
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_predict_batch_matches_predict_and_preserves_ranking(model: VNPR) -> None:
    users = torch.arange(N_USERS)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = torch.stack([model.predict(int(u), items) for u in users])
        out = model.predict_batch(users, items)

    assert out.shape == (N_USERS, N_ITEMS)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)
    r_ref = torch.sort(ref, dim=1, descending=True, stable=True).indices
    r_out = torch.sort(out, dim=1, descending=True, stable=True).indices
    for u, p in (r_ref != r_out).nonzero().tolist():
        gap = abs(ref[u, r_ref[u, p]] - ref[u, r_out[u, p]]).item()
        assert gap < 1e-6, f"ranking swap at a non-tie gap ({gap:.2e})"


def test_predict_batch_on_item_subset_matches_predict(model: VNPR) -> None:
    users = torch.arange(3)
    items = torch.tensor([0, 7, 42, 199])

    with torch.no_grad():
        ref = torch.stack([model.predict(int(u), items) for u in users])
        out = model.predict_batch(users, items)

    assert model._catalogue_visual_cache is None
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)


def test_has_no_mlp_hidden_layers_or_item_bias(model: VNPR) -> None:
    assert not hasattr(model, "mlp")
    assert not hasattr(model, "hidden_layers")
    assert not hasattr(model, "item_bias")
    assert not hasattr(model, "visual_transform")


def test_dense_layer_is_a_single_neuron_over_k_plus_dv(model: VNPR) -> None:
    assert model.dense.out_features == 1
    assert model.dense.in_features == K + DV


def test_l2_penalises_whole_embedding_matrices_and_not_the_dense_layer() -> None:
    model = _make({"l2_reg": 1e-2}).eval()
    users, pos, neg = torch.arange(4), torch.arange(4), torch.arange(4, 8)
    untouched_row = N_ITEMS - 1  # never in pos / neg

    with torch.no_grad():
        loss_before = model.bpr_loss(*model(users, pos, neg)).item()
        model.item_embedding_neg.weight[untouched_row] *= 10.0
        loss_after = model.bpr_loss(*model(users, pos, neg)).item()
        reg_before = model.l2_reg().item()
        model.dense.weight *= 10.0
        reg_after = model.l2_reg().item()

    assert loss_after > loss_before
    assert reg_after == pytest.approx(reg_before)


def test_dropout_perturbs_training_forward_and_is_inert_in_eval() -> None:
    model = _make({"dropout": 0.5})
    users, pos, neg = torch.arange(N_USERS), torch.arange(N_USERS), torch.arange(60, 90)

    model.train()
    with torch.no_grad():
        first, _ = model(users, pos, neg)
        second, _ = model(users, pos, neg)
    model.eval()
    with torch.no_grad():
        eval_first, _ = model(users, pos, neg)
        eval_second, _ = model(users, pos, neg)

    assert not torch.equal(first, second)
    assert torch.equal(eval_first, eval_second)
    assert torch.allclose(eval_first, _manual_branch(model, users, pos, neg=False), atol=1e-6)


def test_online_fusion_scores_all_paths_without_using_the_cache() -> None:
    model = _make(visual=_visual(1, (N_ITEMS, 2, DV))).eval()
    users, pos, neg = torch.arange(5), torch.arange(5), torch.arange(20, 25)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        r_pos, r_neg = model(users, pos, neg)
        ref = torch.stack([model.predict(int(u), items) for u in users])
        out = model.predict_batch(users, items)

    assert model._online_fusion is not None
    assert r_pos.shape == r_neg.shape == (5,)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)
    assert model._catalogue_visual_cache is None


def test_catalogue_cache_is_populated_in_eval_and_invalidated_by_train(model: VNPR) -> None:
    users, items = torch.arange(2), torch.arange(N_ITEMS)

    with torch.no_grad():
        model.predict_batch(users, items)
    assert model._catalogue_visual_cache is not None

    model.train()
    assert model._catalogue_visual_cache is None


def test_state_dict_holds_exactly_the_paper_parameters(model: VNPR) -> None:
    expected = {
        "user_embedding.weight",
        "item_embedding.weight",
        "item_embedding_neg.weight",
        "visual_user_embedding.weight",
        "dense.weight",
        "dense.bias",
    }

    assert set(model.state_dict()) == expected
    assert model.visual_user_embedding.embedding_dim == DV


def test_missing_visual_embeddings_are_rejected() -> None:
    with pytest.raises(RuntimeError):
        VNPR(N_USERS, N_ITEMS, visual_embeddings=None, config={"latent_dim": K})
